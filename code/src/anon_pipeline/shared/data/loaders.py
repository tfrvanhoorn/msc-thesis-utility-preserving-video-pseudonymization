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

try:  # Python 3.7 compatibility
    from typing import Protocol as _Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol as _Protocol


class SupportsDataConfig(_Protocol):
    dataset_path: Path
    dataset_type: str
    options: Optional[Dict[str, Any]]

DEFAULT_IMAGE_PATTERNS: List[str] = ["*.jpg", "*.jpeg", "*.png"]


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
    ) -> None:
        self.root = root
        self.identities = list(identities) if identities else None
        self.patterns = list(patterns) if patterns else list(DEFAULT_IMAGE_PATTERNS)
        self.recursive = recursive
        self.shuffle = shuffle

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
                # No per-identity cap; rely on upstream sampling controls

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
        max_samples: int | None = None,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.images_dir = root / images_subdir
        self.identity_path = root / identity_file
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.max_samples = max_samples
        self.shuffle = shuffle
        self._entries = self._load_identity_entries()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        per_identity_counter: MutableMapping[str, int] = defaultdict(int)
        yielded = 0
        entries = list(self._entries)
        if self.shuffle:
            random.shuffle(entries)
        for filename, identity in entries:
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
            yielded += 1
            if self.max_samples and yielded >= self.max_samples:
                break

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
    )


def _build_celeba_dataset(config: SupportsDataConfig) -> CelebADataset:
    options = config.options or {}
    return CelebADataset(
        root=config.dataset_path,
        images_subdir=options.get("images_subdir", "img_align_celeba"),
        identity_file=options.get("identity_file", "identity_CelebA.txt"),
        identities=_normalize_identities(options.get("identities")),
        max_samples=_as_optional_int(options.get("max_samples")),
        shuffle=bool(options.get("shuffle", False)),
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
    "voxceleb_video": _build_voxceleb_video_dataset,
}


def _build_identity_sample_index(config: SupportsDataConfig, identities: Sequence[str]) -> dict[str, list[SampleReference]]:
    dataset_type = config.dataset_type.lower()
    options = config.options or {}
    index: dict[str, list[SampleReference]] = {}

    if dataset_type == "image_folder":
        patterns = options.get("patterns") or DEFAULT_IMAGE_PATTERNS
        recursive = bool(options.get("recursive", False))
        for identity in identities:
            paths = _collect_image_paths_for_identity(config.dataset_path, identity, patterns, recursive)
            if paths:
                index[identity] = [SampleReference(identity, path, "image", context="static") for path in paths]
        return index

    if dataset_type == "celeba":
        images_subdir = options.get("images_subdir", "img_align_celeba")
        identity_file = options.get("identity_file", "identity_CelebA.txt")
        max_samples = _as_optional_int(options.get("max_samples"))
        identity_map = _collect_celeba_paths(config.dataset_path, identity_file)
        for identity in identities:
            files = identity_map.get(str(identity), [])
            if max_samples:
                files = files[:max_samples]
            if files:
                refs = [SampleReference(identity, config.dataset_path / images_subdir / fname, "image", context="static") for fname in files]
                index[str(identity)] = refs
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
        videos_in_youtube = 0
        for pattern in patterns:
            for video_path in sorted(youtube_dir.glob(pattern)):
                if not video_path.is_file():
                    continue
                videos_seen += 1
                videos_in_youtube += 1
                if max_videos_per_identity and videos_seen > max_videos_per_identity:
                    return refs
                if max_videos_per_youtube_id and videos_in_youtube > max_videos_per_youtube_id:
                    break
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
    """Emit identity-balanced batches with minimal in-memory buffering.

    Identities are iterated in shuffled order per epoch. For each identity we
    select samples and form batches with the first ``batch_identities`` identities.
    When ``group_by_video`` is True, ``samples_per_identity`` counts videos and
    all windows from those videos are included (subject to earlier caps). Otherwise
    it counts individual samples. Only lightweight references are cached; frames
    are loaded just-in-time for each batch.
    """

    def __init__(
        self,
        sample_index: dict[str, list[SampleReference]],
        identity_to_index: dict[str, int],
        *,
        batch_identities: int,
        samples_per_identity: int,
        shuffle_identities: bool = True,
        seed: int | None = None,
        group_by_video: bool = False,
    ) -> None:
        self.sample_index = sample_index
        self.identity_to_index = identity_to_index
        self.batch_identities = batch_identities
        self.samples_per_identity = samples_per_identity
        self.shuffle_identities = shuffle_identities
        self.group_by_video = group_by_video
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

    def _group_refs(self, refs: list[SampleReference]) -> list[list[SampleReference]]:
        if not self.group_by_video:
            return [[r] for r in refs]
        buckets: dict[Path, list[SampleReference]] = {}
        for ref in refs:
            buckets.setdefault(ref.path, []).append(ref)
        groups = list(buckets.values())
        if self.shuffle_identities:
            self._rng.shuffle(groups)
        else:
            groups.sort(key=lambda g: str(g[0].path))
        return groups

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

    def _materialize_batch(self, batch_refs: list[tuple[str, list[SampleReference]]]) -> Optional[dict[str, Any]]:
        batch_sequences: list[torch.Tensor] = []
        batch_labels: list[int] = []
        batch_seq_lens: list[int] = []
        batch_identities: list[str] = []
        batch_contexts: list[str] = []
        batch_sources: list[str] = []

        for identity, refs in batch_refs:
            loaded: list[tuple[torch.Tensor, str, str]] = []
            groups = self._group_refs(refs)
            if self.group_by_video and self.samples_per_identity and len(groups) < self.samples_per_identity:
                continue
            groups = groups[: self.samples_per_identity] if self.samples_per_identity else groups
            for group in groups:
                for ref in group:
                    frames, ctx, source_id = self._load_reference(ref)
                    if frames is None:
                        continue
                    loaded.append((frames, ctx, source_id))
            if len(loaded) < 1:
                continue
            for frames, ctx, source_id in loaded:
                batch_sequences.append(frames)
                batch_labels.append(self.identity_to_index[identity])
                batch_seq_lens.append(int(frames.shape[0]))
                batch_identities.append(identity)
                batch_contexts.append(ctx)
                batch_sources.append(source_id)

        if len(batch_sequences) < self.batch_identities:
            return None

        max_len = max(batch_seq_lens)
        padded = [self._pad_sequence(seq, max_len) for seq in batch_sequences]
        return {
            "frames": torch.stack(padded, dim=0),
            "label": torch.tensor(batch_labels, dtype=torch.long),
            "seq_lens": torch.tensor(batch_seq_lens, dtype=torch.long),
            "identity": batch_identities,
            "context": batch_contexts,
            "source": batch_sources,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        identity_order = self._iter_identity_order()
        batch_refs: list[tuple[str, list[SampleReference]]] = []

        for identity in identity_order:
            refs = self._iter_refs_for_identity(identity)
            if not refs:
                continue
            batch_refs.append((identity, refs))

            if len(batch_refs) == self.batch_identities:
                batch = self._materialize_batch(batch_refs)
                if batch is not None:
                    yield batch
                batch_refs = []

        # Drop any incomplete batch with fewer than batch_identities identities to
        # match the expected projector training shape.
