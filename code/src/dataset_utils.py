from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[0]
PROJECT_ROOT = current_file.parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.prepared import build_prepared_filename  # noqa: E402
from data.video_io import get_video_fps, load_video_frames, write_mp4  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

DEFAULT_VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.avi", "*.mov")


@dataclass(frozen=True)
class SourceVideo:
    identity: str
    youtube_id: str | None
    video_path: Path


def _iter_video_paths(root: Path, patterns: tuple[str, ...] = DEFAULT_VIDEO_PATTERNS, *, recursive: bool = False) -> list[Path]:
    videos: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        videos.extend([p for p in iterator if p.is_file()])
    return sorted(set(videos))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for inference/evaluation naming contract")

    parser.add_argument("--type", type=str, required=True, choices=["voxceleb", "video_folder"], help="Dataset preparation type")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to source dataset root")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save prepared videos")
    parser.add_argument(
        "--video_folder_identity",
        type=str,
        default="video",
        help="Shared identity name used for all videos when --type video_folder",
    )

    parser.add_argument("--max_identities", type=int, default=None, help="Limit number of identities")
    parser.add_argument("--max_videos_per_youtube_id", type=int, default=None, help="Max videos per YouTube ID")
    parser.add_argument("--max_videos_per_id", type=int, default=None, help="Max videos per identity")
    parser.add_argument("--max_frames_per_video", type=int, default=64, help="Maximum sampled frames per video")
    parser.add_argument("--fps", type=float, default=10.0, help="Target frames-per-second for sampled output videos")

    return parser.parse_args()


def _collect_voxceleb_sources(args: argparse.Namespace) -> list[SourceVideo]:
    base = args.data_path / "dev" / "mp4"
    if not base.exists():
        raise FileNotFoundError(f"VoxCeleb path not found: {base}")

    identities = sorted([p.name for p in base.iterdir() if p.is_dir()])
    if args.max_identities is not None:
        identities = identities[: args.max_identities]

    sources: list[SourceVideo] = []
    for identity in identities:
        identity_dir = base / identity
        youtube_dirs = sorted([p for p in identity_dir.iterdir() if p.is_dir()])
        videos_seen_identity = 0

        for youtube_dir in youtube_dirs:
            candidates = _iter_video_paths(youtube_dir)
            if args.max_videos_per_youtube_id is not None:
                candidates = candidates[: args.max_videos_per_youtube_id]

            for video_path in candidates:
                videos_seen_identity += 1
                if args.max_videos_per_id is not None and videos_seen_identity > args.max_videos_per_id:
                    break

                sources.append(
                    SourceVideo(
                        identity=identity,
                        youtube_id=youtube_dir.name,
                        video_path=video_path,
                    )
                )

            if args.max_videos_per_id is not None and videos_seen_identity >= args.max_videos_per_id:
                break

    return sources


def _collect_video_folder_sources(args: argparse.Namespace) -> list[SourceVideo]:
    if not args.data_path.exists():
        raise FileNotFoundError(f"Video folder path not found: {args.data_path}")

    shared_identity = args.video_folder_identity.strip()
    if not shared_identity:
        raise ValueError("--video_folder_identity must be non-empty")
    if "_" in shared_identity:
        raise ValueError("--video_folder_identity cannot contain underscores")

    all_videos = _iter_video_paths(args.data_path, recursive=True)
    if args.max_videos_per_id is not None:
        all_videos = all_videos[: args.max_videos_per_id]

    return [
        SourceVideo(identity=shared_identity, youtube_id=None, video_path=video_path)
        for video_path in all_videos
    ]


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.max_identities is not None and args.max_identities <= 0:
        raise ValueError("--max_identities must be > 0 when provided")
    if args.max_videos_per_youtube_id is not None and args.max_videos_per_youtube_id <= 0:
        raise ValueError("--max_videos_per_youtube_id must be > 0 when provided")
    if args.max_videos_per_id is not None and args.max_videos_per_id <= 0:
        raise ValueError("--max_videos_per_id must be > 0 when provided")
    if args.max_frames_per_video <= 0:
        raise ValueError("--max_frames_per_video must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "voxceleb":
        sources = _collect_voxceleb_sources(args)
    elif args.type == "video_folder":
        sources = _collect_video_folder_sources(args)
    else:
        raise ValueError(f"Unsupported dataset type: {args.type}")

    identity_counts: dict[str, int] = {}
    processed = 0
    skipped = 0
    entries: list[dict[str, object]] = []

    for src in sources:
        sample_index = identity_counts.get(src.identity, 0) + 1
        identity_counts[src.identity] = sample_index

        source_fps = get_video_fps(src.video_path)
        frame_step = max(1, int(round(source_fps / float(args.fps))))
        sampled = load_video_frames(
            src.video_path,
            max_frames=args.max_frames_per_video,
            frame_step=frame_step,
            convert_rgb=True,
        )
        if sampled is None:
            skipped += 1
            continue

        prepared_name = build_prepared_filename(
            src.identity,
            sample_index,
            src.video_path.stem,
            extension=".mp4",
        )
        output_path = output_dir / prepared_name
        frames = [sampled[i] for i in range(sampled.shape[0])]
        codec = write_mp4(output_path, frames, fps=float(args.fps))

        processed += 1
        entries.append(
            {
                "identity": src.identity,
                "youtube_id": src.youtube_id,
                "source_video": str(src.video_path),
                "sample_index": sample_index,
                "prepared_video": str(output_path.relative_to(output_dir).as_posix()),
                "source_fps": float(source_fps),
                "target_fps": float(args.fps),
                "effective_frame_step": int(frame_step),
                "sampled_frames": int(sampled.shape[0]),
                "codec": codec,
            }
        )

    report_path = output_dir / "dataset_preparation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "type": args.type,
                "data_path": str(args.data_path),
                "output_dir": str(output_dir),
                "processed_videos": processed,
                "skipped_videos": skipped,
                "settings": {
                    "max_identities": args.max_identities,
                    "max_videos_per_youtube_id": args.max_videos_per_youtube_id,
                    "max_videos_per_id": args.max_videos_per_id,
                    "max_frames_per_video": args.max_frames_per_video,
                    "fps": args.fps,
                },
                "entries": entries,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Dataset preparation complete | type=%s | processed_videos=%d | skipped_videos=%d | output_dir=%s | report=%s",
        args.type,
        processed,
        skipped,
        output_dir,
        report_path,
    )


if __name__ == "__main__":
    main()
