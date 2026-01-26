import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure src is on sys.path when running directly
PROJECT_SRC = Path(__file__).resolve().parent.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from anon_pipeline.shared.data.loaders import build_dataset
from anon_pipeline.shared.data.splits import build_dataloader_for_identities, split_identities
from anon_pipeline.shared.data.video_loaders import VoxCelebVideoDataset


NUM_IDENTITIES = 2
VIDEOS_PER_IDENTITY = 2
WINDOW_SIZE = 4
WINDOW_STEP = 4
FRAMES_PER_SAMPLE = 12
_WINDOWS_PER_VIDEO = FRAMES_PER_SAMPLE // WINDOW_STEP


class _Config:
    def __init__(self, dataset_path, dataset_type, options=None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.options = options or {}


def _make_voxceleb_root(dataset_root: Path):
    base = dataset_root / "dev" / "mp4"
    video_paths = sorted(base.rglob("*.mp4")) if base.exists() else []
    return base, video_paths


def test_voxceleb_dataset_windows(dataset_root: Path):
    root, video_paths = _make_voxceleb_root(dataset_root)
    dataset = VoxCelebVideoDataset(root=root, window_size=WINDOW_SIZE, window_step=WINDOW_STEP)
    samples = list(dataset)

    print("found videos", len(video_paths))
    print("dataset windows count", len(samples))
    if samples:
        first = samples[0]
        print("first sample path", first.video_path)
        print("first sample start_frame", first.start_frame)
    else:
        print("no windows produced; check frame extraction or codec support")


def test_dataloader_emits_frame_windows(dataset_root: Path):
    root, _ = _make_voxceleb_root(dataset_root)
    cfg = _Config(
        dataset_path=root,
        dataset_type="voxceleb_video",
        options={"window_size": WINDOW_SIZE, "window_step": WINDOW_STEP},
    )

    split = split_identities(cfg, train_fraction=1.0)
    loader = build_dataloader_for_identities(
        cfg, split.train, batch_size=2, shuffle=False, load_images=True
    )

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("dataloader empty; no windows available from videos")
        return

    print("batch keys", list(batch.keys()))
    print("frames shape", batch.get("frames", [np.array([])])[0].shape)
    print("labels", batch.get("label"))

    dataset = build_dataset(cfg)
    print("build_dataset windows count", len(list(dataset)))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]  # .../code
    data_root = repo_root / "data" / "voxceleb"
    if not data_root.exists():
        logging.error("VoxCeleb data not found at %s", data_root)
        return

    logging.info("Using VoxCeleb data at: %s", data_root)
    logging.info("Running dataset window test")
    test_voxceleb_dataset_windows(data_root)
    logging.info("Running dataloader test")
    test_dataloader_emits_frame_windows(data_root)


if __name__ == "__main__":
    main()
