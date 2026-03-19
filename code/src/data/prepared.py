from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_VIDEO_PATTERNS: tuple[str, ...] = ("*.mp4", "*.mkv", "*.avi", "*.mov")
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
