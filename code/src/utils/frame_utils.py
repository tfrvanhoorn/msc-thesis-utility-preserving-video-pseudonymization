from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Avoid shadowing the stdlib logging module with utils/logging.py when run as a script.
_script_dir = Path(__file__).resolve().parent
_removed_script_dir = False
if str(_script_dir) in sys.path:
    sys.path.remove(str(_script_dir))
    _removed_script_dir = True
import logging
if _removed_script_dir:
    sys.path.insert(0, str(_script_dir))

import cv2

logger = logging.getLogger(__name__)


def _iter_mp4_paths(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.mp4") if path.is_file()])


def _build_output_dir(root: Path, output_dir: Path, video_path: Path) -> Path:
    try:
        relative = video_path.relative_to(root)
    except ValueError:
        relative = video_path.name
        return output_dir / Path(relative).with_suffix("")
    return output_dir / relative.with_suffix("")


def _compute_frame_step(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if source_fps <= 0:
        return 1
    step = int(round(source_fps / target_fps))
    return max(step, 1)


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", video_path)
        return 0

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = _compute_frame_step(source_fps, target_fps)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_index = 0
    output_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % frame_step == 0:
                output_path = output_dir / f"frame_{output_index:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    logger.warning("Failed to write frame %s", output_path)
                else:
                    saved += 1
                output_index += 1
            frame_index += 1
    finally:
        cap.release()

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames from mp4 files")
    parser.add_argument("--input-dir", dest="input_dir", type=Path, required=True, help="Root folder with mp4 files")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, required=True, help="Root folder to write frame folders")
    parser.add_argument("--output_dir", dest="output_dir", type=Path, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=float, required=True, help="Target frames-per-second for extracted frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    videos = _iter_mp4_paths(input_dir)
    logger.info("Found %d mp4 files under %s", len(videos), input_dir)

    total_frames = 0
    for video_path in videos:
        dest_dir = _build_output_dir(input_dir, output_dir, video_path)
        saved = extract_frames(video_path, dest_dir, args.fps)
        total_frames += saved
        logger.info("Extracted %d frames from %s", saved, video_path)

    logger.info("Done. Total extracted frames: %d", total_frames)


if __name__ == "__main__":
    main()
