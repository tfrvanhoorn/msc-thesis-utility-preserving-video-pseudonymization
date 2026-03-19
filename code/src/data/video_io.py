from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

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


def get_video_fps(path: Path, fallback_fps: float = 10.0) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            logger.warning("Failed to open video for fps probe: %s", path)
            return float(fallback_fps)
        fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps_val <= 0.0 or not np.isfinite(fps_val):
        return float(fallback_fps)
    return fps_val


def load_video_frames(
    path: Path,
    *,
    max_frames: int | None = None,
    frame_step: int = 1,
    convert_rgb: bool = True,
) -> np.ndarray | None:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", path)
        return None

    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_index % frame_step == 0:
                if convert_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        return None
    return np.stack(frames, axis=0)


def _to_bgr_uint8(frame: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().float().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(f"Expected frame with 3 dimensions, got shape {arr.shape}")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels in last dim, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def write_mp4(
    path: Path,
    frames: list[np.ndarray | torch.Tensor],
    *,
    fps: float = 10.0,
    preferred_codecs: tuple[str, ...] = ("avc1", "mp4v"),
) -> str:
    if not frames:
        raise ValueError("Cannot write empty video")

    path.parent.mkdir(parents=True, exist_ok=True)

    first = _to_bgr_uint8(frames[0])
    height, width = int(first.shape[0]), int(first.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid frame size for video write: {(height, width)}")

    chosen_codec = None
    writer = None
    for codec in preferred_codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        probe = cv2.VideoWriter(str(path), fourcc, float(max(fps, 1e-3)), (width, height))
        if probe.isOpened():
            writer = probe
            chosen_codec = codec
            break
        probe.release()

    if writer is None or chosen_codec is None:
        raise RuntimeError(f"Failed to initialize MP4 writer for {path}")

    try:
        writer.write(first)
        for frame in frames[1:]:
            out = _to_bgr_uint8(frame)
            if out.shape[0] != height or out.shape[1] != width:
                out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(out)
    finally:
        writer.release()

    return chosen_codec


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
