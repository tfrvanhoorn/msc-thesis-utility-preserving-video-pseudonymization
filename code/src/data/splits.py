from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .loaders import (
    SupportsDataConfig,
    IdentityBatchingDataset,
    _build_identity_sample_index,
    build_dataset,
    unified_video_collate_fn,
)
from .prepared import DEFAULT_PREPARED_REGEX, collect_prepared_images, compile_prepared_regex


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

    if dataset_type == "prepared_images":
        options = config.options or {}
        regex = compile_prepared_regex(options.get("prepared_filename_regex", DEFAULT_PREPARED_REGEX))
        refs = collect_prepared_images(config.dataset_path, regex)
        return sorted({ref.identity for ref in refs})

    if dataset_type == "voxceleb_video":
        base = Path(config.dataset_path) / "dev" / "mp4"
        if not base.exists():
            return []
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    if dataset_type == "video_folder":
        if not config.dataset_path.exists():
            return []
        identities: set[str] = set()
        for path in config.dataset_path.glob("*.mp4"):
            if not path.is_file():
                continue
            stem = path.stem
            identity = stem.split("_", 1)[0] if "_" in stem else stem
            identities.add(identity)
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


def _config_with_identities(config: SupportsDataConfig, identities: Sequence[str], *, shuffle: bool = False):
    options = dict(config.options or {})
    options["identities"] = list(identities)
    # Propagate shuffle preference through options so IterableDatasets can randomize internally
    options["shuffle"] = shuffle or bool(getattr(config, "options", {}).get("shuffle", False))

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
    identity_batching: bool = False,
    group_by_video: bool = False,
    max_samples_per_identity: int | None = None,
) -> DataLoader:
    identity_to_index = {ident: idx for idx, ident in enumerate(identities)}

    # Propagate per-identity caps into options for datasets and batch index builders
    if max_samples_per_identity is not None:
        options = dict(config.options or {})
        options["max_samples_per_identity"] = max_samples_per_identity
        if config.dataset_type.lower() in {"voxceleb_video", "video_folder"}:
            options.setdefault("max_videos_per_identity", max_samples_per_identity)

        class _ConfigProxy:
            def __init__(self, base: SupportsDataConfig, opts):
                self.dataset_path = base.dataset_path
                self.dataset_type = base.dataset_type
                self.options = opts

        config = _ConfigProxy(config, options)

    if identity_batching:
        if config.dataset_type.lower() != "prepared_images":
            raise ValueError("identity_batching is currently only supported for dataset_type='prepared_images'")
        sample_index = _build_identity_sample_index(config, identities)
        batched_dataset = IdentityBatchingDataset(
            sample_index,
            identity_to_index,
            shuffle_identities=shuffle,
        )
        return DataLoader(batched_dataset, batch_size=None, shuffle=False, num_workers=num_workers)

    cfg = _config_with_identities(config, identities, shuffle=shuffle)
    dataset = build_dataset(cfg)

    # IterableDataset does not support DataLoader-level shuffling; randomization can be handled inside the dataset
    effective_collate = collate_fn or (lambda batch: unified_video_collate_fn(batch, identity_to_index=identity_to_index))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=effective_collate)


def build_train_test_loaders(
    config: SupportsDataConfig,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
    max_identities: int | None = None,
    max_samples_per_identity: int | None = None,
    batch_size: int = 4,
    identity_batching: bool = False,
    group_by_video: bool = False,
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
        identity_batching=identity_batching,
        group_by_video=group_by_video,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        max_samples_per_identity=max_samples_per_identity,
    )
    test_loader = build_dataloader_for_identities(
        config,
        split.test,
        batch_size=batch_size,
        identity_batching=identity_batching,
        group_by_video=group_by_video,
        shuffle=shuffle_test,
        num_workers=num_workers,
        collate_fn=collate_fn,
        max_samples_per_identity=max_samples_per_identity,
    )
    return split, train_loader, test_loader

