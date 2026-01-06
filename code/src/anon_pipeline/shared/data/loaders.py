from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, MutableMapping, Sequence

from ...config import DataConfig

DEFAULT_IMAGE_PATTERNS: List[str] = ["*.jpg", "*.jpeg", "*.png"]


@dataclass(frozen=True)
class ImageSample:
    identity: str
    path: Path


class ImageFolderDataset(Iterable[ImageSample]):
    def __init__(
        self,
        root: Path,
        identities: Sequence[str] | None = None,
        max_per_identity: int | None = None,
        patterns: Sequence[str] | None = None,
        recursive: bool = False,
    ) -> None:
        self.root = root
        self.identities = list(identities) if identities else None
        self.max_per_identity = max_per_identity
        self.patterns = list(patterns) if patterns else list(DEFAULT_IMAGE_PATTERNS)
        self.recursive = recursive

    def __iter__(self) -> Iterator[ImageSample]:
        identities = self.identities or self._discover_identities()
        for identity in identities:
            identity_dir = self.root / identity
            if not identity_dir.exists():
                continue
            count = 0
            for image_path in self._iter_image_paths(identity_dir):
                yield ImageSample(identity=identity, path=image_path)
                count += 1
                if self.max_per_identity and count >= self.max_per_identity:
                    break

    def _discover_identities(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def _iter_image_paths(self, identity_dir: Path) -> Iterator[Path]:
        candidates: set[Path] = set()
        for pattern in self.patterns:
            globber = identity_dir.rglob(pattern) if self.recursive else identity_dir.glob(pattern)
            for path in globber:
                if path.is_file():
                    candidates.add(path)
        for path in sorted(candidates):
            yield path


class CelebADataset(Iterable[ImageSample]):
    def __init__(
        self,
        root: Path,
        images_subdir: str = "img_align_celeba",
        identity_file: str = "identity_CelebA.txt",
        identities: Sequence[str] | None = None,
        max_per_identity: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = root
        self.images_dir = root / images_subdir
        self.identity_path = root / identity_file
        self.identities = set(str(identity) for identity in identities) if identities else None
        self.max_per_identity = max_per_identity
        self.max_samples = max_samples
        self._entries = self._load_identity_entries()

    def __iter__(self) -> Iterator[ImageSample]:
        per_identity_counter: MutableMapping[str, int] = defaultdict(int)
        yielded = 0
        for filename, identity in self._entries:
            if self.identities and identity not in self.identities:
                continue
            if self.max_per_identity and per_identity_counter[identity] >= self.max_per_identity:
                continue
            image_path = self.images_dir / filename
            if not image_path.exists():
                continue
            yield ImageSample(identity=identity, path=image_path)
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


def build_dataset(config: DataConfig) -> Iterable[ImageSample]:
    dataset_type = config.dataset_type.lower()
    builder = _DATASET_BUILDERS.get(dataset_type)
    if builder is None:
        known = ", ".join(sorted(_DATASET_BUILDERS))
        raise ValueError(f"Unsupported dataset type '{config.dataset_type}'. Known types: {known}")
    dataset_root = config.dataset_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    return builder(config)


def iter_samples(config: DataConfig) -> Iterator[ImageSample]:
    dataset = build_dataset(config)
    yield from dataset


def _build_image_folder_dataset(config: DataConfig) -> ImageFolderDataset:
    options = config.options or {}
    return ImageFolderDataset(
        root=config.dataset_path,
        identities=_normalize_identities(options.get("identities")),
        max_per_identity=_as_optional_int(options.get("max_per_identity")),
        patterns=options.get("patterns"),
        recursive=bool(options.get("recursive", False)),
    )


def _build_celeba_dataset(config: DataConfig) -> CelebADataset:
    options = config.options or {}
    return CelebADataset(
        root=config.dataset_path,
        images_subdir=options.get("images_subdir", "img_align_celeba"),
        identity_file=options.get("identity_file", "identity_CelebA.txt"),
        identities=_normalize_identities(options.get("identities")),
        max_per_identity=_as_optional_int(options.get("max_per_identity")),
        max_samples=_as_optional_int(options.get("max_samples")),
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


_DATASET_BUILDERS: Dict[str, Callable[[DataConfig], Iterable[ImageSample]]] = {
    "image_folder": _build_image_folder_dataset,
    "celeba": _build_celeba_dataset,
}
