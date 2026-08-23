#!/usr/bin/env python3
"""Build and audit a zero-copy RL-VO view over the split TartanAir subsets."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Sequence


MANIFEST_NAMES = (
    "easy_train_gray_all/manifest.json",
    "easy_val_gray_all/manifest.json",
    "hard_train_gray_all/manifest.json",
    "hard_val_gray_all/manifest.json",
)

EXPECTED = {
    "train_trajectories": 337,
    "train_frames": 279987,
    "val_trajectories": 32,
    "val_frames": 26650,
    "total_trajectories": 369,
    "total_frames": 306637,
}


def load_test_split(loader_file: Path) -> List[str]:
    module = ast.parse(loader_file.read_text(encoding="utf-8"), filename=str(loader_file))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "test_split" for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("test_split is not a list of strings")
            return value
    raise ValueError(f"test_split not found in {loader_file}")


def load_records(source_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for relative_manifest in MANIFEST_NAMES:
        manifest_path = source_root / relative_manifest
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = str(payload["split"])
        for trajectory in payload["trajectories"]:
            record = dict(trajectory)
            record["split"] = split
            record["manifest"] = str(manifest_path)
            records.append(record)
    return records


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting symlink: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise


def prepare_view(source_root: Path, view_root: Path, records: Sequence[Dict[str, object]]) -> None:
    view_root.mkdir(parents=True, exist_ok=True)
    calibration = source_root / "easy_train_gray_all/calibration/tartan_pinhole.yaml"
    _safe_symlink(calibration, view_root / "calibration/tartan_pinhole.yaml")
    for record in records:
        source = Path(str(record["output"]))
        trajectory = Path(str(record["trajectory"]))
        _safe_symlink(source, view_root / trajectory)


def _count_pose_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def audit(
    source_root: Path,
    view_root: Path,
    loader_file: Path,
    records: Sequence[Dict[str, object]],
    full: bool,
) -> Dict[str, object]:
    names = [str(record["trajectory"]) for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate trajectory names in manifests")

    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    test_split = load_test_split(loader_file)
    val_names = sorted(str(record["trajectory"]) for record in val_records)
    if val_names != sorted(test_split):
        raise RuntimeError("manifest validation trajectories differ from official test_split")
    if set(test_split).intersection(str(record["trajectory"]) for record in train_records):
        raise RuntimeError("official validation trajectory leaked into training records")

    stats = {
        "train_trajectories": len(train_records),
        "train_frames": sum(int(record["frames"]) for record in train_records),
        "val_trajectories": len(val_records),
        "val_frames": sum(int(record["frames"]) for record in val_records),
        "total_trajectories": len(records),
        "total_frames": sum(int(record["frames"]) for record in records),
    }
    if stats != EXPECTED:
        raise RuntimeError(f"unexpected inventory: {stats}")

    checked_frames = 0
    for record in records:
        expected_frames = int(record["frames"])
        if int(record["converted"]) != expected_frames or int(record["skipped"]) != 0:
            raise RuntimeError(f"incomplete conversion record: {record['trajectory']}")
        source = Path(str(record["output"]))
        image_dir = source / "image_left_gray"
        pose_file = source / "pose_left.txt"
        last_image = image_dir / f"{expected_frames - 1:06d}_left.jpg"
        if not image_dir.is_dir() or not pose_file.is_file():
            raise RuntimeError(f"missing image or pose input: {record['trajectory']}")
        if not (image_dir / "000000_left.jpg").is_file() or not last_image.is_file():
            raise RuntimeError(f"non-contiguous endpoint filenames: {record['trajectory']}")
        if full:
            image_count = sum(1 for path in image_dir.iterdir() if path.name.endswith("_left.jpg"))
            pose_count = _count_pose_rows(pose_file)
            if image_count != expected_frames or pose_count != expected_frames:
                raise RuntimeError(
                    f"count mismatch for {record['trajectory']}: "
                    f"images={image_count}, poses={pose_count}, expected={expected_frames}"
                )
            checked_frames += image_count

        destination = view_root / str(record["trajectory"])
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise RuntimeError(f"invalid view link: {destination}")

    calibration = view_root / "calibration/tartan_pinhole.yaml"
    if not calibration.is_symlink() or not calibration.is_file():
        raise RuntimeError("missing calibration link")

    view_trajectories = sorted(
        str(path.relative_to(view_root)) for path in view_root.glob("*/*/*") if path.is_dir()
    )
    if view_trajectories != sorted(names):
        raise RuntimeError("unified view contains an unexpected trajectory set")

    return {
        "status": "passed",
        "source_root": str(source_root.resolve()),
        "view_root": str(view_root.resolve()),
        "loader_file": str(loader_file.resolve()),
        "full_file_count_check": full,
        "checked_frames": checked_frames,
        **stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--loader-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.source_root)
    if args.command == "prepare":
        prepare_view(args.source_root, args.view_root, records)
    payload = audit(args.source_root, args.view_root, args.loader_file, records, args.full)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
