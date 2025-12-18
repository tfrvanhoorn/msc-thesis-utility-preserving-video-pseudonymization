from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..config import ExperimentConfig


def _resolve_path(value: Optional[Path], base_dir: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def load_config_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_dict(payload)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    config = build_config(load_config_payload(config_path))
    config_dir = config_path.parent
    project_root = config_dir.parent if config_dir.parent != config_dir else config_dir

    config.data.dataset_path = _resolve_path(config.data.dataset_path, project_root)
    config.data.cache_dir = _resolve_path(config.data.cache_dir, project_root)
    config.detector.root = _resolve_path(config.detector.root, project_root)
    config.embedding.root = _resolve_path(config.embedding.root, project_root)

    return config
