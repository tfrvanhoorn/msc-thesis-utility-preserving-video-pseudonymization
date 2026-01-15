from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass
class DataConfig:
    dataset_path: Path
    cache_dir: Optional[Path] = None
    dataset_type: str = "image_folder"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    score_threshold: float = 0.6
    image_size: int = 160
    margin: int = 0
    min_face_size: Optional[int] = 20
    max_faces: Optional[int] = None
    device: Optional[str] = None


@dataclass
class FeatureSelectorConfig:
    keep: Sequence[str] = field(default_factory=list)


@dataclass
class EmbeddingConfig:
    method: str = "facenet"
    pretrained: str = "vggface2"
    embedding_size: int = 512
    device: Optional[str] = None
    feature_selector: FeatureSelectorConfig = field(default_factory=FeatureSelectorConfig)


@dataclass
class QuantizationConfig:
    method: str = "auto"
    num_prototypes: int = 64
    max_iters: int = 20
    tol: float = 1e-4
    random_seed: Optional[int] = 123
    train_split: float = 0.0
    max_train_samples: Optional[int] = None
    output_dim: int = 2048
    input_dim: Optional[int] = None


@dataclass
class SeedConfig:
    secret_key: str
    digest: str = "sha256"


@dataclass
class QuantizationExperimentConfig:
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantizationExperimentConfig":
        data_cfg = payload.get("data", {})
        detector_cfg = cls._require(payload, "detector")
        embedding_cfg = cls._require(payload, "embedding")
        quantization_cfg = cls._require(payload, "quantization")
        seed_cfg = cls._require(payload, "seed")

        def _path_or_none(value: Optional[str]) -> Optional[Path]:
            return Path(value) if value else None

        max_train_raw = quantization_cfg.get("max_train_samples")
        quantization_dict = {
            "method": quantization_cfg.get("method", "auto"),
            "num_prototypes": quantization_cfg.get("num_prototypes", 64),
            "max_iters": quantization_cfg.get("max_iters", 20),
            "tol": quantization_cfg.get("tol", 1e-4),
            "random_seed": quantization_cfg.get("random_seed", 123),
            "train_split": float(quantization_cfg.get("train_split", 0.0) or 0.0),
            "max_train_samples": int(max_train_raw) if max_train_raw not in (None, "") else None,
            "output_dim": int(quantization_cfg.get("output_dim", 2048)),
            "input_dim": int(quantization_cfg["input_dim"]) if "input_dim" in quantization_cfg else None,
        }

        feature_selector_cfg = FeatureSelectorConfig(**embedding_cfg.get("feature_selector", {}))
        embedding_kwargs = {
            **embedding_cfg,
            "method": embedding_cfg.get("method") or embedding_cfg.get("type") or "facenet",
            "pretrained": embedding_cfg.get("pretrained", "vggface2"),
            "device": embedding_cfg.get("device"),
            "feature_selector": feature_selector_cfg,
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
                    "device": detector_cfg.get("device"),
                }
            ),
            embedding=EmbeddingConfig(**embedding_kwargs),
            quantization=QuantizationConfig(**quantization_dict),
            seed=SeedConfig(**seed_cfg),
        )
