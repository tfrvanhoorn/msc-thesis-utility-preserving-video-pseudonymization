from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parent.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from anon_pipeline.shared.data import loaders as loaders_module
from anon_pipeline.shared.data.loaders import build_dataset
from anon_pipeline.shared.data.splits import build_dataloader_for_identities, list_identities


class _Config:
    def __init__(self, dataset_path: Path, dataset_type: str, options: dict | None = None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.options = options or {}


def _write_fake_video_files(root: Path) -> None:
    (root / "alice_001.mp4").write_bytes(b"fake")
    (root / "alice_002.mp4").write_bytes(b"fake")
    (root / "bob_001.mp4").write_bytes(b"fake")
    (root / "ignore.txt").write_text("not-a-video", encoding="utf-8")


def _patch_video_io(monkeypatch) -> None:
    def _fake_count(_path: Path) -> int:
        return 8

    def _fake_load(_path: Path, _start: int, window_size: int, frame_stride: int = 1):
        del frame_stride
        return np.zeros((window_size, 4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(loaders_module, "get_video_frame_count", _fake_count)
    monkeypatch.setattr(loaders_module, "load_video_window", _fake_load)


def test_video_folder_list_identities(tmp_path: Path) -> None:
    _write_fake_video_files(tmp_path)
    cfg = _Config(dataset_path=tmp_path, dataset_type="video_folder")

    identities = list_identities(cfg)

    assert identities == ["alice", "bob"]


def test_video_folder_dataset_iteration(tmp_path: Path, monkeypatch) -> None:
    _write_fake_video_files(tmp_path)
    _patch_video_io(monkeypatch)
    cfg = _Config(
        dataset_path=tmp_path,
        dataset_type="video_folder",
        options={
            "window_size": 4,
            "window_step": 4,
            "max_windows_per_video": 1,
        },
    )

    dataset = build_dataset(cfg)
    samples = list(dataset)

    assert len(samples) == 3
    assert all(sample["identity"] in {"alice", "bob"} for sample in samples)
    assert all(sample["frames"].shape == (4, 3, 4, 4) for sample in samples)


def test_video_folder_identity_batching_loader(tmp_path: Path, monkeypatch) -> None:
    _write_fake_video_files(tmp_path)
    _patch_video_io(monkeypatch)
    cfg = _Config(
        dataset_path=tmp_path,
        dataset_type="video_folder",
        options={
            "window_size": 4,
            "window_step": 4,
            "max_windows_per_video": 1,
            "max_videos_per_identity": 1,
        },
    )

    loader = build_dataloader_for_identities(
        cfg,
        identities=["alice", "bob"],
        batch_size=2,
        identity_batching=True,
        batch_identities=2,
        samples_per_identity=1,
        group_by_video=True,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert torch.is_tensor(batch["frames"])
    assert batch["frames"].shape[0] == 2
    assert set(batch["identity"]) == {"alice", "bob"}
    assert batch["label"].tolist() == [0, 1]
