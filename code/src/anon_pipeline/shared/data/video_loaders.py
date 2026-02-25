from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset

from .video_io import get_video_frame_count, load_video_window

DEFAULT_VIDEO_PATTERNS: list[str] = ["*.mp4", "*.mkv", "*.avi", "*.mov"]


@dataclass(frozen=True)
class VideoWindowSample:
    identity: str
    video_path: Path
    start_frame: int
    window_size: int
    frame_stride: int = 1


class VoxCelebVideoDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        max_videos_per_identity: int | None = None,
        max_videos_per_youtube_id: int | None = None,
        min_youtube_id_per_identity: int | None = None,
        window_size: int = 16,
        frame_stride: int = 1,
        window_step: int | None = None,
        patterns: Sequence[str] | None = None,
        max_windows_per_video: int | None = None,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.base_dir = root / "dev" / "mp4"
        self.identities = list(identities) if identities else None
        self.max_videos_per_identity = max_videos_per_identity
        self.max_videos_per_youtube_id = max_videos_per_youtube_id
        self.min_youtube_id_per_identity = min_youtube_id_per_identity
        self.window_size = window_size
        self.frame_stride = frame_stride
        self.window_step = window_step if window_step is not None else window_size * frame_stride
        self.patterns = list(patterns) if patterns else list(DEFAULT_VIDEO_PATTERNS)
        self.shuffle = shuffle
        self.max_windows_per_video = max_windows_per_video
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.window_step <= 0:
            raise ValueError("window_step must be positive")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        identities = self.identities or self._discover_identities()
        if self.shuffle:
            identities = list(identities)
            random.shuffle(identities)
        for identity in identities:
            identity_dir = self.base_dir / identity
            if not identity_dir.exists():
                continue
            youtube_dirs = [p for p in identity_dir.iterdir() if p.is_dir()]
            if self.min_youtube_id_per_identity is not None and len(youtube_dirs) < self.min_youtube_id_per_identity:
                continue
            if self.shuffle:
                random.shuffle(youtube_dirs)
            else:
                youtube_dirs.sort()
            videos_seen = 0
            for youtube_dir in youtube_dirs:
                videos_seen_in_youtube = 0
                for video_path in self._iter_video_paths(youtube_dir):
                    videos_seen += 1
                    videos_seen_in_youtube += 1
                    if self.max_videos_per_identity and videos_seen > self.max_videos_per_identity:
                        break
                    if self.max_videos_per_youtube_id and videos_seen_in_youtube > self.max_videos_per_youtube_id:
                        break

                    total_frames = get_video_frame_count(video_path)
                    usable_start_limit = total_frames - (self.window_size - 1) * self.frame_stride
                    if usable_start_limit <= 0:
                        continue
                    starts = self._iter_starts(usable_start_limit)
                    windows_from_video = 0
                    for start in starts:
                        window = load_video_window(
                            video_path,
                            start,
                            self.window_size,
                            frame_stride=self.frame_stride,
                        )
                        if window is None:
                            continue

                        frames = torch.from_numpy(window).permute(0, 3, 1, 2).float() / 255.0
                        youtube_id = video_path.parent.name
                        source_id = str(video_path.relative_to(self.base_dir))
                        yield {
                            "frames": frames,
                            "identity": identity,
                            "context": youtube_id,
                            "source": source_id,
                        }
                        windows_from_video += 1
                        if self.max_windows_per_video and windows_from_video >= self.max_windows_per_video:
                            break
                if self.max_videos_per_identity and videos_seen >= self.max_videos_per_identity:
                    break

    def _discover_identities(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted([p.name for p in self.base_dir.iterdir() if p.is_dir()])

    def _iter_video_paths(self, youtube_dir: Path) -> Iterator[Path]:
        candidates: list[Path] = []
        for pattern in self.patterns:
            for path in youtube_dir.glob(pattern):
                if path.is_file():
                    candidates.append(path)
        if self.shuffle:
            random.shuffle(candidates)
        else:
            candidates.sort()
        for path in candidates:
            yield path

    def _iter_starts(self, start_limit: int) -> Iterator[int]:
        if self.window_step <= 0:
            raise ValueError("window_step must be positive")
        for start in range(0, max(0, start_limit), self.window_step):
            yield start
