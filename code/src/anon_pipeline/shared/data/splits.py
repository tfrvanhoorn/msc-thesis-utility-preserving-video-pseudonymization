from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .loaders import ImageSample, SupportsDataConfig, build_dataset


@dataclass(frozen=True)
class IdentitySplit:
    train: List[str]
    test: List[str]


def list_identities(config: SupportsDataConfig) -> List[str]:
    dataset_type = config.dataset_type.lower()

    if dataset_type == "image_folder":
        return sorted([p.name for p in config.dataset_path.iterdir() if p.is_dir()])

    if dataset_type == "celeba":
        options = config.options or {}
        identity_file = options.get("identity_file", "identity_CelebA.txt")
        identity_path = Path(config.dataset_path) / identity_file
        identities: set[str] = set()
        with identity_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                _, identity = stripped.split()
                identities.add(str(identity))
        return sorted(identities)

    raise ValueError(f"Unsupported dataset type '{config.dataset_type}' for identity listing")


def split_identities(
    config: SupportsDataConfig,
    train_fraction: float = 0.8,
    seed: int = 0,
    max_identities: int | None = None,
) -> IdentitySplit:
    identities = list_identities(config)
    rng = random.Random(seed)
    rng.shuffle(identities)

    if max_identities is not None:
        identities = identities[:max_identities]

    cutoff = int(len(identities) * train_fraction)
    cutoff = max(1, min(cutoff, len(identities) - 1)) if len(identities) > 1 else len(identities)

    train_ids = identities[:cutoff]
    test_ids = identities[cutoff:]
    return IdentitySplit(train=train_ids, test=test_ids)


def _config_with_identities(config: SupportsDataConfig, identities: Sequence[str]):
    options = dict(config.options or {})
    options["identities"] = list(identities)

    class _ConfigProxy:
        def __init__(self, base: SupportsDataConfig, opts):
            self.dataset_path = base.dataset_path
            self.dataset_type = base.dataset_type
            self.options = opts

    return _ConfigProxy(config, options)


def build_dataloader_for_identities(
    config: SupportsDataConfig,
    identities: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
    load_images: bool = True,
) -> DataLoader:
    cfg = _config_with_identities(config, identities)
    dataset = _LoadedDictDataset(build_dataset(cfg)) if load_images else build_dataset(cfg)

    if collate_fn is None:
        # Default collate: stack dicts into lists/tensors where possible
        def _default_collate(batch):
            images: list[np.ndarray] = []
            labels: list[int] = []
            for item in batch:
                if not isinstance(item, dict):
                    continue
                img = item.get("image")
                lbl = item.get("label")
                if img is None or lbl is None:
                    continue
                images.append(img)
                labels.append(int(lbl))
            if not images:
                return {"image": [], "label": torch.tensor([], dtype=torch.long)}
            return {"image": images, "label": torch.tensor(labels, dtype=torch.long)}

        collate_fn = _default_collate

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


def build_train_test_loaders(
    config: SupportsDataConfig,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
    max_identities: int | None = None,
    batch_size: int = 4,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    num_workers: int = 0,
    collate_fn=None,
) -> Tuple[IdentitySplit, DataLoader, DataLoader]:
    split = split_identities(config, train_fraction=train_fraction, seed=seed, max_identities=max_identities)
    train_loader = build_dataloader_for_identities(
        config,
        split.train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = build_dataloader_for_identities(
        config,
        split.test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return split, train_loader, test_loader


class _LoadedDictDataset:
    def __init__(self, iterable: Iterable[ImageSample]):
        self.iterable = iterable
        self._cache: list[dict[str, object]] | None = None

    def __iter__(self) -> Iterator[dict[str, object]]:
        if self._cache is not None:
            yield from self._cache
            return

        self._cache = []
        for sample in self.iterable:
            try:
                arr = np.asarray(Image.open(sample.path).convert("RGB"))
            except Exception:
                continue
            item = {"image": arr, "label": int(sample.identity)}
            self._cache.append(item)
            yield item

    def __len__(self) -> int:
        if self._cache is None:
            # Materialize once to populate cache and length
            list(iter(self))
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, object]:
        if self._cache is None:
            list(iter(self))
        return self._cache[idx]

