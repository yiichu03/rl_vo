from pathlib import Path

from experiments.tartanair_official.dataset_view import EXPECTED, load_test_split


ROOT = Path(__file__).resolve().parents[1]


def test_official_test_split_contract():
    split = load_test_split(ROOT / "dataloader/tartan_loader.py")
    assert len(split) == EXPECTED["val_trajectories"] == 32
    assert len(set(split)) == len(split)
    assert "abandonedfactory/Easy/P011" in split
    assert "westerndesert/Hard/P007" in split


def test_preregistered_inventory_is_internally_consistent():
    assert EXPECTED["train_trajectories"] + EXPECTED["val_trajectories"] == EXPECTED["total_trajectories"]
    assert EXPECTED["train_frames"] + EXPECTED["val_frames"] == EXPECTED["total_frames"]
