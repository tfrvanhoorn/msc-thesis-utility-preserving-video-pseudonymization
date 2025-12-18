from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass
class DataConfig:
    dataset_path: Path
    cache_dir: Optional[Path] = None
    dataset_type: str = "image_folder"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    name: str = "buffalo_l"
    release_name: Optional[str] = None
    score_threshold: float = 0.95
    det_size: Tuple[int, int] = (640, 640)
    max_faces: Optional[int] = None
    ctx_id: int = -1
    providers: Sequence[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    root: Optional[Path] = None


@dataclass
class EmbeddingConfig:
    name: str = "arcface_r100_v1"
    release_name: Optional[str] = None
    model_name: Optional[str] = None
    embedding_size: int = 512
    ctx_id: int = -1
    providers: Sequence[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    root: Optional[Path] = None

@dataclass
class QuantizationConfig:
    num_subspaces: int = 32
    num_prototypes: int = 64
    max_iters: int = 20
    tol: float = 1e-4
    random_seed: Optional[int] = 123
    output_mode: str = "majority"
    train_split: float = 0.0
    max_train_samples: Optional[int] = None


@dataclass
class SeedConfig:
    secret_key: str
    digest: str = "sha256"


@dataclass
class ExperimentConfig:
    data: DataConfig
    detector: DetectorConfig
    embedding: EmbeddingConfig
    quantization: QuantizationConfig
    seed: SeedConfig

    @staticmethod
    def _require(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise KeyError(f"Missing required config key: {key}")
        return mapping[key]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        data_cfg = payload.get("data", {})
        detector_cfg = cls._require(payload, "detector")
        embedding_cfg = cls._require(payload, "embedding")
        quantization_cfg = cls._require(payload, "quantization")
        seed_cfg = cls._require(payload, "seed")

        def _path_or_none(value: Optional[str]) -> Optional[Path]:
            return Path(value) if value else None

        max_train_raw = quantization_cfg.get("max_train_samples")
        quantization_dict = {
            "num_subspaces": quantization_cfg.get("num_subspaces", 32),
            "num_prototypes": quantization_cfg.get("num_prototypes", 64),
            "max_iters": quantization_cfg.get("max_iters", 20),
            "tol": quantization_cfg.get("tol", 1e-4),
            "random_seed": quantization_cfg.get("random_seed", 123),
            "output_mode": quantization_cfg.get("output_mode", "majority"),
            "train_split": float(quantization_cfg.get("train_split", 0.0) or 0.0),
            "max_train_samples": int(max_train_raw) if max_train_raw not in (None, "") else None,
        }

        return cls(
            data=DataConfig(
                dataset_path=Path(cls._require(data_cfg, "dataset_path")),
                cache_dir=_path_or_none(data_cfg.get("cache_dir")),
                dataset_type=data_cfg.get("dataset_type", "image_folder"),
                options=dict(data_cfg.get("options", {})),
            ),
            detector=DetectorConfig(
                **{
                    **detector_cfg,
                    "root": _path_or_none(detector_cfg.get("root")),
                    "release_name": detector_cfg.get("release_name"),
                }
            ),
            embedding=EmbeddingConfig(
                **{
                    **embedding_cfg,
                    "root": _path_or_none(embedding_cfg.get("root")),
                    "release_name": embedding_cfg.get("release_name"),
                    "model_name": embedding_cfg.get("model_name") or embedding_cfg.get("name"),
                }
            ),
            quantization=QuantizationConfig(**quantization_dict),
            seed=SeedConfig(**seed_cfg),
        )
