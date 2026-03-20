from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from .video_loaders import VoxCelebVideoDataset, DEFAULT_VIDEO_PATTERNS
from .video_io import get_video_frame_count, load_video_window
from .prepared import (
    DEFAULT_PREPARED_REGEX,
    collect_prepared_images,
    compile_prepared_regex,
)

try:  # Python 3.7 compatibility
    from typing import Protocol as _Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol as _Protocol


class SupportsDataConfig(_Protocol):
    dataset_path: Path
    dataset_type: str
    options: Optional[Dict[str, Any]]

DEFAULT_IMAGE_PATTERNS: List[str] = ["*.jpg", "*.jpeg", "*.png"]
DEFAULT_VIDEO_FOLDER_PATTERNS: List[str] = ["*.mp4"]


@dataclass(frozen=True)
class ImageSample:
    identity: str
    path: Path


@dataclass(frozen=True)
class SampleReference:
    """Lightweight pointer to a sample without keeping frames in memory."""

    identity: str
    path: Path
    kind: str  # "image" or "video_window"
    start: int | None = None
    window_size: int | None = None
    frame_stride: int | None = None
    context: str | None = None
    source: str | None = None


class ImageFolderDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        patterns: Sequence[str] | None = None,
        recursive: bool = False,
        shuffle: bool = False,
        max_samples_per_identity: int | None = None,
    ) -> None:
        self.root = root
        self.identities = list(identities) if identities else None
        self.patterns = list(patterns) if patterns else list(DEFAULT_IMAGE_PATTERNS)
        self.recursive = recursive
        self.shuffle = shuffle
        self.max_samples_per_identity = max_samples_per_identity

    def __iter__(self) -> Iterator[dict[str, Any]]:
        identities = self.identities or self._discover_identities()
        if self.shuffle:
            identities = list(identities)
            random.shuffle(identities)
        for identity in identities:
            identity_dir = self.root / identity
            if not identity_dir.exists():
                continue
            count = 0
            for image_path in self._iter_image_paths(identity_dir):
                frame = _load_image_tensor(image_path)
                if frame is None:
                    continue
                yield {
                    "frames": frame.unsqueeze(0),  # Seq len 1 for static images
                    "identity": identity,
                    "context": "static",
                }
                count += 1
                if self.max_samples_per_identity is not None and count >= self.max_samples_per_identity:
                    break

    def _discover_identities(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def _iter_image_paths(self, identity_dir: Path) -> Iterator[Path]:
        candidates: list[Path] = []
        for pattern in self.patterns:
            globber = identity_dir.rglob(pattern) if self.recursive else identity_dir.glob(pattern)
            for path in globber:
                if path.is_file():
                    candidates.append(path)
        if self.shuffle:
            random.shuffle(candidates)
        else:
            candidates.sort()
        for path in candidates:
            yield path


class CelebADataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        images_subdir: str = "img_align_celeba",
        identity_file: str = "identity_CelebA.txt",
        identities: Sequence[str] | None = None,
        max_samples_per_identity: int | None = None,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.images_dir = root / images_subdir
        self.identity_path = root / identity_file
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.max_samples_per_identity = max_samples_per_identity
        self.shuffle = shuffle
        self._entries = self._load_identity_entries()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        per_identity_counter: MutableMapping[str, int] = defaultdict(int)
        entries = list(self._entries)
        if self.shuffle:
            random.shuffle(entries)
        for filename, identity in entries:
            if self.max_samples_per_identity and per_identity_counter[identity] >= self.max_samples_per_identity:
                continue
            if self.identities and identity not in self.identities:
                continue
            image_path = self.images_dir / filename
            if not image_path.exists():
                continue
            frame = _load_image_tensor(image_path)
            if frame is None:
                continue
            yield {
                "frames": frame.unsqueeze(0),
                "identity": identity,
                "context": "static",
            }
            per_identity_counter[identity] += 1

    def _load_identity_entries(self) -> List[tuple[str, str]]:
        entries: List[tuple[str, str]] = []
        with self.identity_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                filename, identity = stripped.split()
                entries.append((filename, str(identity)))
        return entries


class PreparedImagesDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        prepared_filename_regex: str = DEFAULT_PREPARED_REGEX,
        shuffle: bool = False,
        max_samples_per_identity: int | None = None,
    ) -> None:
        self.root = root
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.prepared_filename_regex = prepared_filename_regex
        self.shuffle = shuffle
        self.max_samples_per_identity = max_samples_per_identity

    def __iter__(self) -> Iterator[dict[str, Any]]:
        regex = compile_prepared_regex(self.prepared_filename_regex)
        refs = collect_prepared_images(self.root, regex)
        if self.shuffle:
            random.shuffle(refs)

        per_identity_counter: MutableMapping[str, int] = defaultdict(int)
        for ref in refs:
            if self.identities and ref.identity not in self.identities:
                continue
            if self.max_samples_per_identity and per_identity_counter[ref.identity] >= self.max_samples_per_identity:
                continue

            frame = _load_image_tensor(ref.image_path)
            if frame is None:
                continue

            yield {
                "frames": frame.unsqueeze(0),
                "identity": ref.identity,
                "context": "static",
                "source": str(ref.image_path.name),
            }
            per_identity_counter[ref.identity] += 1


class VideoFolderDataset(IterableDataset):
    """Load video windows from a flat folder of video files.

    Identity is derived from the file stem prefix before the first underscore.
    Example: ``id123_clip01.mp4`` -> identity ``id123``.
    """

    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        patterns: Sequence[str] | None = None,
        max_videos_per_identity: int | None = None,
        window_size: int = 16,
        frame_stride: int = 1,
        window_step: int | None = None,
        max_windows_per_video: int | None = None,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.patterns = list(patterns) if patterns else list(DEFAULT_VIDEO_FOLDER_PATTERNS)
        self.max_videos_per_identity = max_videos_per_identity
        self.window_size = window_size
        self.frame_stride = frame_stride
        self.window_step = window_step if window_step is not None else window_size * frame_stride
        self.max_windows_per_video = max_windows_per_video
        self.shuffle = shuffle
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.window_step <= 0:
            raise ValueError("window_step must be positive")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        video_records = self._collect_video_records()
        per_identity_seen: dict[str, int] = defaultdict(int)

        for identity, video_path in video_records:
            if self.max_videos_per_identity and per_identity_seen[identity] >= self.max_videos_per_identity:
                continue
            per_identity_seen[identity] += 1

            total_frames = get_video_frame_count(video_path)
            usable_start_limit = total_frames - (self.window_size - 1) * self.frame_stride
            if usable_start_limit <= 0:
                continue

            windows_from_video = 0
            for start in range(0, max(0, usable_start_limit), self.window_step):
                window = load_video_window(
                    video_path,
                    start,
                    self.window_size,
                    frame_stride=self.frame_stride,
                )
                if window is None:
                    continue

                frames = torch.from_numpy(window).permute(0, 3, 1, 2).float() / 255.0
                yield {
                    "frames": frames,
                    "identity": identity,
                    "context": video_path.stem,
                    "source": str(video_path.name),
                }

                windows_from_video += 1
                if self.max_windows_per_video and windows_from_video >= self.max_windows_per_video:
                    break

    def _collect_video_records(self) -> list[tuple[str, Path]]:
        records: list[tuple[str, Path]] = []
        for pattern in self.patterns:
            for path in self.root.glob(pattern):
                if not path.is_file():
                    continue
                identity = _video_folder_identity_from_path(path)
                if self.identities is not None and identity not in self.identities:
                    continue
                records.append((identity, path))

        if self.shuffle:
            random.shuffle(records)
        else:
            records.sort(key=lambda item: str(item[1]))
        return records


def build_dataset(config: SupportsDataConfig) -> Iterable:
    dataset_type = config.dataset_type.lower()
    builder = _DATASET_BUILDERS.get(dataset_type)
    if builder is None:
        known = ", ".join(sorted(_DATASET_BUILDERS))
        raise ValueError(f"Unsupported dataset type '{config.dataset_type}'. Known types: {known}")
    dataset_root = config.dataset_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    return builder(config)


def iter_samples(config: SupportsDataConfig) -> Iterator:
    dataset = build_dataset(config)
    yield from dataset


def _build_image_folder_dataset(config: SupportsDataConfig) -> ImageFolderDataset:
    options = config.options or {}
    return ImageFolderDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        patterns=options.get("patterns"),
        recursive=bool(options.get("recursive", False)),
        shuffle=bool(options.get("shuffle", False)),
        max_samples_per_identity=_as_optional_int(options.get("max_samples_per_identity")),
    )


def _build_celeba_dataset(config: SupportsDataConfig) -> CelebADataset:
    options = config.options or {}
    return CelebADataset(
        root=config.dataset_path,
        images_subdir=options.get("images_subdir", "img_align_celeba"),
        identity_file=options.get("identity_file", "identity_CelebA.txt"),
        identities=_normalize_identities(options.get("identities")),
        max_samples_per_identity=_as_optional_int(options.get("max_samples_per_identity")),
        shuffle=bool(options.get("shuffle", False)),
    )


def _build_prepared_images_dataset(config: SupportsDataConfig) -> PreparedImagesDataset:
    options = config.options or {}
    return PreparedImagesDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        prepared_filename_regex=options.get("prepared_filename_regex", DEFAULT_PREPARED_REGEX),
        shuffle=bool(options.get("shuffle", False)),
        max_samples_per_identity=_as_optional_int(options.get("max_samples_per_identity")),
    )


def _build_voxceleb_video_dataset(config: SupportsDataConfig) -> VoxCelebVideoDataset:
    options = config.options or {}
    return VoxCelebVideoDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        max_videos_per_identity=_as_optional_int(options.get("max_videos_per_identity")),
        max_videos_per_youtube_id=_as_optional_int(options.get("max_videos_per_youtube_id")),
        min_youtube_id_per_identity=_as_optional_int(options.get("min_youtube_id_per_identity")),
        window_size=_as_optional_int(options.get("window_size")) or 16,
        frame_stride=_as_optional_int(options.get("frame_stride")) or 1,
        window_step=_as_optional_int(options.get("window_step")),
        patterns=options.get("patterns"),
        max_windows_per_video=_as_optional_int(options.get("max_windows_per_video")),
        shuffle=bool(options.get("shuffle", False)),
    )


def _build_video_folder_dataset(config: SupportsDataConfig) -> VideoFolderDataset:
    options = config.options or {}
    max_videos = _as_optional_int(options.get("max_videos_per_identity"))
    if max_videos is None:
        max_videos = _as_optional_int(options.get("max_samples_per_identity"))
    return VideoFolderDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        patterns=options.get("patterns") or DEFAULT_VIDEO_FOLDER_PATTERNS,
        max_videos_per_identity=max_videos,
        window_size=_as_optional_int(options.get("window_size")) or 16,
        frame_stride=_as_optional_int(options.get("frame_stride")) or 1,
        window_step=_as_optional_int(options.get("window_step")),
        max_windows_per_video=_as_optional_int(options.get("max_windows_per_video")),
        shuffle=bool(options.get("shuffle", False)),
    )


def _normalize_identities(raw: Sequence[Any] | str | int | None) -> Sequence[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, int):
        return [str(raw)]
    return [str(identity) for identity in raw]


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


_DATASET_BUILDERS: Dict[str, Callable[[SupportsDataConfig], Iterable]] = {
    "image_folder": _build_image_folder_dataset,
    "celeba": _build_celeba_dataset,
    "prepared_images": _build_prepared_images_dataset,
    "voxceleb_video": _build_voxceleb_video_dataset,
    "video_folder": _build_video_folder_dataset,
}


def _build_identity_sample_index(config: SupportsDataConfig, identities: Sequence[str]) -> dict[str, list[SampleReference]]:
    dataset_type = config.dataset_type.lower()
    options = config.options or {}
    index: dict[str, list[SampleReference]] = {}

    if dataset_type == "image_folder":
        patterns = options.get("patterns") or DEFAULT_IMAGE_PATTERNS
        recursive = bool(options.get("recursive", False))
        max_per_identity = _as_optional_int(options.get("max_samples_per_identity"))
        for identity in identities:
            paths = _collect_image_paths_for_identity(config.dataset_path, identity, patterns, recursive)
            if max_per_identity:
                paths = paths[:max_per_identity]
            if paths:
                index[identity] = [SampleReference(identity, path, "image", context="static") for path in paths]
        return index

    if dataset_type == "celeba":
        images_subdir = options.get("images_subdir", "img_align_celeba")
        identity_file = options.get("identity_file", "identity_CelebA.txt")
        max_samples = _as_optional_int(options.get("max_samples_per_identity"))
        identity_map = _collect_celeba_paths(config.dataset_path, identity_file)
        for identity in identities:
            files = identity_map.get(str(identity), [])
            if max_samples:
                files = files[:max_samples]
            if files:
                refs = [SampleReference(identity, config.dataset_path / images_subdir / fname, "image", context="static") for fname in files]
                index[str(identity)] = refs
        return index

    if dataset_type == "prepared_images":
        max_samples = _as_optional_int(options.get("max_samples_per_identity"))
        regex = compile_prepared_regex(options.get("prepared_filename_regex", DEFAULT_PREPARED_REGEX))
        refs = collect_prepared_images(config.dataset_path, regex)

        grouped: dict[str, list[SampleReference]] = defaultdict(list)
        wanted_identities = {str(identity) for identity in identities}
        for ref in refs:
            if ref.identity not in wanted_identities:
                continue
            grouped[ref.identity].append(
                SampleReference(
                    identity=ref.identity,
                    path=ref.image_path,
                    kind="image",
                    context="static",
                    source=str(ref.image_path.name),
                )
            )

        for identity in identities:
            identity_key = str(identity)
            identity_refs = grouped.get(identity_key, [])
            if max_samples:
                identity_refs = identity_refs[:max_samples]
            if identity_refs:
                index[identity_key] = identity_refs

        return index

    if dataset_type == "voxceleb_video":
        base = config.dataset_path / "dev" / "mp4"
        window_size = _as_optional_int(options.get("window_size")) or 16
        frame_stride = _as_optional_int(options.get("frame_stride")) or 1
        window_step = options.get("window_step")
        if window_step is None:
            window_step = window_size * frame_stride
        window_step = int(window_step)
        patterns = options.get("patterns") or DEFAULT_VIDEO_PATTERNS
        max_windows_per_video = _as_optional_int(options.get("max_windows_per_video"))
        max_videos_per_identity = _as_optional_int(options.get("max_videos_per_identity"))
        max_videos_per_youtube_id = _as_optional_int(options.get("max_videos_per_youtube_id"))
        min_youtube_id_per_identity = _as_optional_int(options.get("min_youtube_id_per_identity"))
        for identity in identities:
            refs = _collect_voxceleb_windows_for_identity(
                base,
                str(identity),
                patterns,
                window_size,
                frame_stride,
                window_step,
                max_videos_per_identity,
                max_videos_per_youtube_id,
                min_youtube_id_per_identity,
                max_windows_per_video,
            )
            if refs:
                index[str(identity)] = refs
        return index

    if dataset_type == "video_folder":
        window_size = _as_optional_int(options.get("window_size")) or 16
        frame_stride = _as_optional_int(options.get("frame_stride")) or 1
        window_step = options.get("window_step")
        if window_step is None:
            window_step = window_size * frame_stride
        window_step = int(window_step)
        patterns = options.get("patterns") or DEFAULT_VIDEO_FOLDER_PATTERNS
        max_windows_per_video = _as_optional_int(options.get("max_windows_per_video"))
        max_videos_per_identity = _as_optional_int(options.get("max_videos_per_identity"))
        if max_videos_per_identity is None:
            max_videos_per_identity = _as_optional_int(options.get("max_samples_per_identity"))

        for identity in identities:
            refs = _collect_video_folder_windows_for_identity(
                config.dataset_path,
                str(identity),
                patterns,
                window_size,
                frame_stride,
                window_step,
                max_videos_per_identity,
                max_windows_per_video,
            )
            if refs:
                index[str(identity)] = refs
        return index

    raise ValueError(f"Unsupported dataset type '{dataset_type}' for identity batching")


def _collect_image_paths_for_identity(root: Path, identity: str, patterns: Sequence[str], recursive: bool) -> list[Path]:
    identity_dir = root / identity
    if not identity_dir.exists():
        return []
    candidates: list[Path] = []
    for pattern in patterns:
        globber = identity_dir.rglob(pattern) if recursive else identity_dir.glob(pattern)
        for path in globber:
            if path.is_file():
                candidates.append(path)
    candidates.sort()
    return candidates


def _collect_celeba_paths(root: Path, identity_file: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    identity_path = root / identity_file
    if not identity_path.exists():
        return {}
    with identity_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            filename, identity = stripped.split()
            mapping[str(identity)].append(filename)
    for files in mapping.values():
        files.sort()
    return mapping


def _collect_voxceleb_windows_for_identity(
    base_dir: Path,
    identity: str,
    patterns: Sequence[str],
    window_size: int,
    frame_stride: int,
    window_step: int,
    max_videos_per_identity: int | None,
    max_videos_per_youtube_id: int | None,
    min_youtube_id_per_identity: int | None,
    max_windows_per_video: int | None,
) -> list[SampleReference]:
    identity_dir = base_dir / identity
    if not identity_dir.exists():
        return []
    youtube_dirs = [p for p in identity_dir.iterdir() if p.is_dir()]
    if min_youtube_id_per_identity is not None and len(youtube_dirs) < min_youtube_id_per_identity:
        return []
    youtube_dirs.sort()

    refs: list[SampleReference] = []
    videos_seen = 0
    for youtube_dir in youtube_dirs:
        video_candidates: list[Path] = []
        for pattern in patterns:
            video_candidates.extend([p for p in youtube_dir.glob(pattern) if p.is_file()])
        video_candidates = sorted(video_candidates)
        if max_videos_per_youtube_id:
            video_candidates = video_candidates[:max_videos_per_youtube_id]

        for video_path in video_candidates:
            videos_seen += 1
            if max_videos_per_identity and videos_seen > max_videos_per_identity:
                return refs

            total_frames = get_video_frame_count(video_path)
            usable_start_limit = total_frames - (window_size - 1) * frame_stride
            if usable_start_limit <= 0:
                continue
            windows_from_video = 0
            for start in range(0, max(0, usable_start_limit), window_step):
                refs.append(
                    SampleReference(
                        identity=identity,
                        path=video_path,
                        kind="video_window",
                        start=start,
                        window_size=window_size,
                        frame_stride=frame_stride,
                        context=youtube_dir.name,
                        source=str(video_path.relative_to(base_dir)),
                    )
                )
                windows_from_video += 1
                if max_windows_per_video and windows_from_video >= max_windows_per_video:
                    break
        if max_videos_per_identity and videos_seen >= max_videos_per_identity:
            return refs
    return refs


def _collect_video_folder_windows_for_identity(
    base_dir: Path,
    identity: str,
    patterns: Sequence[str],
    window_size: int,
    frame_stride: int,
    window_step: int,
    max_videos_per_identity: int | None,
    max_windows_per_video: int | None,
) -> list[SampleReference]:
    video_candidates: list[Path] = []
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file() and _video_folder_identity_from_path(path) == identity:
                video_candidates.append(path)
    video_candidates = sorted(video_candidates)
    if max_videos_per_identity:
        video_candidates = video_candidates[:max_videos_per_identity]

    refs: list[SampleReference] = []
    for video_path in video_candidates:
        total_frames = get_video_frame_count(video_path)
        usable_start_limit = total_frames - (window_size - 1) * frame_stride
        if usable_start_limit <= 0:
            continue
        windows_from_video = 0
        for start in range(0, max(0, usable_start_limit), window_step):
            refs.append(
                SampleReference(
                    identity=identity,
                    path=video_path,
                    kind="video_window",
                    start=start,
                    window_size=window_size,
                    frame_stride=frame_stride,
                    context=video_path.stem,
                    source=str(video_path.name),
                )
            )
            windows_from_video += 1
            if max_windows_per_video and windows_from_video >= max_windows_per_video:
                break
    return refs


def _video_folder_identity_from_path(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[0]


def _load_image_tensor(path: Path) -> torch.Tensor | None:
    try:
        image = Image.open(path).convert("RGB")
        arr = np.array(image, copy=True)
    except Exception:
        return None

    tensor = torch.from_numpy(arr)
    if tensor.dim() != 3 or tensor.shape[-1] != 3:
        return None
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    return tensor


def unified_video_collate_fn(
    batch: Sequence[dict[str, Any]],
    *,
    identity_to_index: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not batch:
        return {
            "frames": torch.empty(0, 0, 0, 0, 0),
            "identity": [],
            "context": [],
            "source": [],
            "label": torch.empty(0, dtype=torch.long),
            "seq_lens": torch.empty(0, dtype=torch.long),
        }

    frames_list: list[torch.Tensor] = []
    identities: list[str] = []
    contexts: list[str] = []
    sources: list[str] = []
    lengths: list[int] = []

    for sample in batch:
        frames = sample.get("frames")
        if frames is None:
            continue
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        if frames.dim() == 3:
            # Single image without sequence dim -> add seq len 1
            frames = frames.unsqueeze(0)
        elif frames.dim() != 4:
            raise ValueError(f"Expected frames with shape (Seq,C,H,W), got {tuple(frames.shape)}")
        lengths.append(int(frames.shape[0]))
        frames_list.append(frames)
        identities.append(str(sample.get("identity", "")))
        contexts.append(str(sample.get("context", "")))
        sources.append(str(sample.get("source", "")))

    if not frames_list:
        return {
            "frames": torch.empty(0, 0, 0, 0, 0),
            "identity": identities,
            "context": contexts,
            "source": sources,
            "label": torch.empty(0, dtype=torch.long),
            "seq_lens": torch.empty(0, dtype=torch.long),
        }

    max_seq_len = max(lengths)
    padded_frames: list[torch.Tensor] = []
    for frames in frames_list:
        if frames.shape[0] < max_seq_len:
            pad_len = max_seq_len - frames.shape[0]
            pad_frame = frames[-1:].expand(pad_len, -1, -1, -1)
            frames = torch.cat([frames, pad_frame], dim=0)
        padded_frames.append(frames)

    batch_frames = torch.stack(padded_frames, dim=0)

    if identity_to_index is None:
        identity_to_index = {}

    labels_list: list[int] = []
    for ident in identities:
        if ident not in identity_to_index:
            identity_to_index[ident] = len(identity_to_index)
        labels_list.append(identity_to_index[ident])

    labels = torch.tensor(labels_list, dtype=torch.long)

    return {
        "frames": batch_frames,
        "identity": identities,
        "context": contexts,
        "source": sources,
        "label": labels,
        "seq_lens": torch.tensor(lengths, dtype=torch.long),
    }


class IdentityBatchingDataset(IterableDataset):
    """Emit no-reuse tuple batches for CelebA-style projector training.

    Each yielded batch contains exactly three source samples arranged as:
    two samples from identity A and one sample from identity B. The training
    pipeline then maps those to the loss tuple
    (A1_key1, A1_key2, A2_key1, B1_key1).
    """

    def __init__(
        self,
        sample_index: dict[str, list[SampleReference]],
        identity_to_index: dict[str, int],
        *,
        shuffle_identities: bool = True,
        seed: int | None = None,
    ) -> None:
        self.sample_index = sample_index
        self.identity_to_index = identity_to_index
        self.shuffle_identities = shuffle_identities
        self._rng = random.Random(seed)

    def _pad_sequence(self, seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.shape[0] >= target_len:
            return seq
        pad_len = target_len - seq.shape[0]
        pad_frame = seq[-1:].expand(pad_len, -1, -1, -1)
        return torch.cat([seq, pad_frame], dim=0)

    def _iter_identity_order(self) -> list[str]:
        identities = list(self.sample_index.keys())
        if self.shuffle_identities:
            self._rng.shuffle(identities)
        else:
            identities.sort()
        return identities

    def _iter_refs_for_identity(self, identity: str) -> list[SampleReference]:
        refs = list(self.sample_index.get(identity, []))
        if self.shuffle_identities:
            self._rng.shuffle(refs)
        return refs

    def _load_reference(self, ref: SampleReference) -> tuple[torch.Tensor | None, str, str]:
        source_id = ref.source or str(ref.path)
        if ref.kind == "image":
            tensor = _load_image_tensor(ref.path)
            if tensor is None:
                return None, ref.context or "", source_id
            return tensor.unsqueeze(0), ref.context or "static", source_id

        if ref.kind == "video_window":
            try:
                window = load_video_window(
                    ref.path,
                    int(ref.start or 0),
                    int(ref.window_size or 0),
                    frame_stride=int(ref.frame_stride or 1),
                )
            except Exception:
                return None, ref.context or "", source_id
            if window is None:
                return None, ref.context or "", source_id
            frames = torch.from_numpy(window).permute(0, 3, 1, 2).float() / 255.0
            return frames, ref.context or "", source_id

        return None, ref.context or "", source_id

    def _build_tensor_batch(
        self,
        loaded_samples: list[tuple[str, torch.Tensor, str, str]],
    ) -> Optional[dict[str, Any]]:
        if not loaded_samples:
            return None

        batch_seq_lens = [int(frames.shape[0]) for _, frames, _, _ in loaded_samples]
        max_len = max(batch_seq_lens)

        padded = [self._pad_sequence(frames, max_len) for _, frames, _, _ in loaded_samples]
        identities = [identity for identity, _, _, _ in loaded_samples]
        contexts = [ctx for _, _, ctx, _ in loaded_samples]
        sources = [source_id for _, _, _, source_id in loaded_samples]
        labels = [self.identity_to_index[identity] for identity in identities]

        return {
            "frames": torch.stack(padded, dim=0),
            "label": torch.tensor(labels, dtype=torch.long),
            "seq_lens": torch.tensor(batch_seq_lens, dtype=torch.long),
            "identity": identities,
            "context": contexts,
            "source": sources,
        }

    def _emit_tuple_batches(self) -> Iterator[dict[str, Any]]:
        # Build an epoch-local pool and consume references immediately on use so
        # no sample can be reused in the same epoch.
        identity_order = self._iter_identity_order()
        pools: dict[str, list[SampleReference]] = {}
        for identity in identity_order:
            refs = self._iter_refs_for_identity(identity)
            if refs:
                pools[identity] = refs

        while True:
            candidates_a = [identity for identity, refs in pools.items() if len(refs) >= 2]
            if not candidates_a:
                break

            if self.shuffle_identities:
                a_identity = self._rng.choice(candidates_a)
            else:
                a_identity = sorted(candidates_a)[0]

            candidates_b = [identity for identity, refs in pools.items() if identity != a_identity and len(refs) >= 1]
            if not candidates_b:
                break

            if self.shuffle_identities:
                b_identity = self._rng.choice(candidates_b)
            else:
                b_identity = sorted(candidates_b)[0]

            requested = [(a_identity, 2), (b_identity, 1)]
            loaded_samples: list[tuple[str, torch.Tensor, str, str]] = []
            missing_required = False

            for identity, needed in requested:
                taken = 0
                while taken < needed:
                    refs = pools.get(identity, [])
                    if not refs:
                        missing_required = True
                        break

                    # Consume once popped, regardless of load success.
                    ref = refs.pop()
                    frames, ctx, source_id = self._load_reference(ref)
                    if frames is None:
                        continue
                    loaded_samples.append((identity, frames, ctx, source_id))
                    taken += 1

                if missing_required:
                    break

            # Keep only identities that still have remaining references.
            pools = {identity: refs for identity, refs in pools.items() if refs}

            if missing_required:
                continue

            batch = self._build_tensor_batch(loaded_samples)
            if batch is not None and int(batch["frames"].shape[0]) == 3:
                yield batch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._emit_tuple_batches()
