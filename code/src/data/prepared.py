from __future__ import annotations

import fnmatch
import logging
import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

DEFAULT_VIDEO_PATTERNS: tuple[str, ...] = ("*.mp4", "*.mkv", "*.avi", "*.mov")
DEFAULT_IMAGE_PATTERNS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")
DEFAULT_PREPARED_REGEX = r"^(?P<identity>[^_]+)_sample(?P<sample>\d+)_(?P<original>.+)$"


@dataclass(frozen=True)
class PreparedVideoRef:
    identity: str
    sample_index: int
    original_name: str
    video_path: Path

    @property
    def key(self) -> tuple[str, int]:
        return (self.identity, self.sample_index)


class PreparedNameError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedImageRef:
    identity: str
    sample_index: int
    original_name: str
    image_path: Path

    @property
    def key(self) -> tuple[str, int]:
        return (self.identity, self.sample_index)


def compile_prepared_regex(pattern: str | None = None) -> re.Pattern[str]:
    compiled = re.compile(pattern or DEFAULT_PREPARED_REGEX)
    required_groups = {"identity", "sample", "original"}
    missing = required_groups - set(compiled.groupindex.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise PreparedNameError(f"Prepared filename regex must define named groups: {missing_list}")
    return compiled


def parse_prepared_video_path(path: Path, regex: re.Pattern[str]) -> PreparedVideoRef:
    match = regex.match(path.stem)
    if match is None:
        raise PreparedNameError(f"Filename does not match prepared naming convention: {path.name}")

    identity = match.group("identity").strip()
    sample_raw = match.group("sample").strip()
    original_name = match.group("original").strip()

    if not identity:
        raise PreparedNameError(f"Prepared filename has empty identity: {path.name}")
    if "_" in identity:
        raise PreparedNameError(f"Identity cannot contain underscores in prepared naming: {path.name}")

    try:
        sample_index = int(sample_raw)
    except ValueError as exc:
        raise PreparedNameError(f"Sample index is not an integer in prepared filename: {path.name}") from exc

    if sample_index <= 0:
        raise PreparedNameError(f"Sample index must be >= 1 in prepared filename: {path.name}")
    if not original_name:
        raise PreparedNameError(f"Prepared filename has empty original token: {path.name}")

    return PreparedVideoRef(
        identity=identity,
        sample_index=sample_index,
        original_name=original_name,
        video_path=path,
    )


def parse_prepared_image_path(path: Path, regex: re.Pattern[str]) -> PreparedImageRef:
    match = regex.match(path.stem)
    if match is None:
        raise PreparedNameError(f"Filename does not match prepared naming convention: {path.name}")

    identity = match.group("identity").strip()
    sample_raw = match.group("sample").strip()
    original_name = match.group("original").strip()

    if not identity:
        raise PreparedNameError(f"Prepared filename has empty identity: {path.name}")
    if "_" in identity:
        raise PreparedNameError(f"Identity cannot contain underscores in prepared naming: {path.name}")

    try:
        sample_index = int(sample_raw)
    except ValueError as exc:
        raise PreparedNameError(f"Sample index is not an integer in prepared filename: {path.name}") from exc

    if sample_index <= 0:
        raise PreparedNameError(f"Sample index must be >= 1 in prepared filename: {path.name}")
    if not original_name:
        raise PreparedNameError(f"Prepared filename has empty original token: {path.name}")

    return PreparedImageRef(
        identity=identity,
        sample_index=sample_index,
        original_name=original_name,
        image_path=path,
    )


def build_prepared_filename(identity: str, sample_index: int, original_name: str, extension: str = ".mp4") -> str:
    identity_clean = identity.strip()
    if not identity_clean:
        raise ValueError("identity must be non-empty")
    if "_" in identity_clean:
        raise ValueError("identity cannot contain underscores in prepared naming")
    if sample_index <= 0:
        raise ValueError("sample_index must be >= 1")

    original_clean = original_name.strip()
    if not original_clean:
        raise ValueError("original_name must be non-empty")

    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{identity_clean}_sample{sample_index}_{original_clean}{ext}"


def iter_video_paths(root: Path, patterns: tuple[str, ...] = DEFAULT_VIDEO_PATTERNS) -> list[Path]:
    videos: list[Path] = []
    for pattern in patterns:
        videos.extend([p for p in root.rglob(pattern) if p.is_file()])
    return sorted(set(videos))


def collect_prepared_videos(
    root: Path,
    regex: re.Pattern[str],
    patterns: tuple[str, ...] = DEFAULT_VIDEO_PATTERNS,
) -> list[PreparedVideoRef]:
    refs: list[PreparedVideoRef] = []
    seen: set[tuple[str, int]] = set()

    for path in iter_video_paths(root, patterns=patterns):
        ref = parse_prepared_video_path(path, regex)
        if ref.key in seen:
            raise PreparedNameError(
                "Duplicate prepared sample key found "
                f"for identity={ref.identity} sample={ref.sample_index}: {path}"
            )
        seen.add(ref.key)
        refs.append(ref)

    return refs


def map_prepared_videos_by_key(
    refs: Iterable[PreparedVideoRef],
) -> dict[tuple[str, int], PreparedVideoRef]:
    mapping: dict[tuple[str, int], PreparedVideoRef] = {}
    for ref in refs:
        if ref.key in mapping:
            raise PreparedNameError(
                "Duplicate prepared sample key found "
                f"for identity={ref.identity} sample={ref.sample_index}: {ref.video_path}"
            )
        mapping[ref.key] = ref
    return mapping


def iter_image_paths(root: Path, patterns: tuple[str, ...] = DEFAULT_IMAGE_PATTERNS) -> Iterator[Path]:
    normalized_patterns = tuple(pattern.lower() for pattern in patterns)

    # Assume input filenames are already grouped/sorted by identity; avoid extra sorting work.
    for dirpath, dirnames, filenames in os.walk(root):
        base_path = Path(dirpath)
        for filename in filenames:
            name_lower = filename.lower()
            if not any(fnmatch.fnmatch(name_lower, pattern) for pattern in normalized_patterns):
                continue
            yield base_path / filename


def collect_prepared_images(
    root: Path,
    regex: re.Pattern[str],
    patterns: tuple[str, ...] = DEFAULT_IMAGE_PATTERNS,
    *,
    identities: set[str] | None = None,
    max_identities: int | None = None,
    stop_after_max_identities: bool = False,
) -> list[PreparedImageRef]:
    refs: list[PreparedImageRef] = []
    seen: set[tuple[str, int]] = set()
    selected_identities: set[str] = set()
    accepted_identities: set[str] = set()
    current_identity: str | None = None

    if max_identities is not None and max_identities <= 0:
        raise ValueError("max_identities must be > 0 when provided")

    wanted = set(identities) if identities is not None else None
    identity_total = len(wanted) if wanted is not None else max_identities

    logging.info(
        "Prepared image scan start | root=%s | max_identities=%s | explicit_identities=%s | stop_after_cap=%s",
        root,
        max_identities,
        len(wanted) if wanted is not None else 0,
        stop_after_max_identities,
    )

    files_scanned = 0
    early_stop_triggered = False

    identity_bar = tqdm(
        total=identity_total,
        desc="Loading identities",
        unit="identity",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    sample_bar = tqdm(
        desc="Loading samples",
        unit="sample",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        for path in iter_image_paths(root, patterns=patterns):
            files_scanned += 1
            sample_bar.set_postfix(
                {
                    "files": files_scanned,
                    "ids": len(accepted_identities),
                },
                refresh=False,
            )

            ref = parse_prepared_image_path(path, regex)

            if wanted is not None and ref.identity not in wanted:
                continue

            if wanted is None and max_identities is not None:
                # With grouped identities, a new identity after reaching cap means we can stop early.
                if ref.identity != current_identity and ref.identity not in selected_identities:
                    if len(selected_identities) >= max_identities:
                        if stop_after_max_identities:
                            early_stop_triggered = True
                            break
                        continue
                    selected_identities.add(ref.identity)
                current_identity = ref.identity

            if ref.key in seen:
                raise PreparedNameError(
                    "Duplicate prepared sample key found "
                    f"for identity={ref.identity} sample={ref.sample_index}: {path}"
                )
            seen.add(ref.key)
            refs.append(ref)
            sample_bar.update(1)

            if ref.identity not in accepted_identities:
                accepted_identities.add(ref.identity)
                identity_bar.update(1)

    finally:
        identity_bar.close()
        sample_bar.close()

    logging.info(
        "Prepared image scan complete | root=%s | files_scanned=%d | accepted_samples=%d | accepted_identities=%d | early_stop=%s",
        root,
        files_scanned,
        len(refs),
        len(accepted_identities),
        early_stop_triggered,
    )

    return refs
