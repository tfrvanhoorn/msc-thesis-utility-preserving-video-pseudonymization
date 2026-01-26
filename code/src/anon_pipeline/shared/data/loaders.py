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

from .video_loaders import VoxCelebVideoDataset

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


class ImageFolderDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        max_per_identity: int | None = None,
        patterns: Sequence[str] | None = None,
        recursive: bool = False,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.identities = list(identities) if identities else None
        self.max_per_identity = max_per_identity
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
                if self.max_per_identity and count >= self.max_per_identity:
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
        max_per_identity: int | None = None,
        max_samples: int | None = None,
        shuffle: bool = False,
    ) -> None:
        self.root = root
        self.images_dir = root / images_subdir
        self.identity_path = root / identity_file
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.max_per_identity = max_per_identity
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
            if self.max_per_identity and per_identity_counter[identity] >= self.max_per_identity:
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
        max_per_identity=_as_optional_int(options.get("max_per_identity")),
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
        max_per_identity=_as_optional_int(options.get("max_per_identity")),
        max_samples=_as_optional_int(options.get("max_samples")),
        shuffle=bool(options.get("shuffle", False)),
    )


def _build_voxceleb_video_dataset(config: SupportsDataConfig) -> VoxCelebVideoDataset:
    options = config.options or {}
    return VoxCelebVideoDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        max_per_identity=_as_optional_int(options.get("max_per_identity")),
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
            "label": torch.empty(0, dtype=torch.long),
            "seq_lens": torch.empty(0, dtype=torch.long),
        }

    frames_list: list[torch.Tensor] = []
    identities: list[str] = []
    contexts: list[str] = []
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

    if not frames_list:
        return {
            "frames": torch.empty(0, 0, 0, 0, 0),
            "identity": identities,
            "context": contexts,
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
        "label": labels,
        "seq_lens": torch.tensor(lengths, dtype=torch.long),
    }
