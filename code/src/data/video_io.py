from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            logger.warning("Failed to open video for frame count: %s", path)
            return 0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return count


def load_video_window(
    path: Path,
    start_frame: int,
    window_size: int,
    frame_stride: int = 1,
    convert_rgb: bool = True,
) -> np.ndarray | None:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", path)
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []

    try:
        for _ in range(window_size):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if frame_stride > 1:
                for _ in range(frame_stride - 1):
                    cap.grab()
    finally:
        cap.release()

    if len(frames) != window_size:
        logger.debug(
            "Insufficient frames for window: path=%s start=%s size=%s got=%s",
            path,
            start_frame,
            window_size,
            len(frames),
        )
        return None

    array = np.stack(frames, axis=0)
    logger.debug(
        "Loaded video window %s start=%s size=%s stride=%s shape=%s dtype=%s",
        path,
        start_frame,
        window_size,
        frame_stride,
        array.shape,
        array.dtype,
    )
    return array
