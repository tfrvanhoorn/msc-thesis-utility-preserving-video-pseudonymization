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
    score_threshold: float = 0.55
    image_size: int = 256
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
class SeedConfig:
    secret_key: str
    digest: str = "sha256"


@dataclass
class ProjectorConfig:
    type: str = "mlp"
    key_dim: int = 128
    hidden_dims: Tuple[int, ...] = (1024, 512)
    dropout: float = 0.0
    enable_input_l2_norm: bool = True
    enable_key_upscaler: bool = True

    def normalized_type(self) -> str:
        proj_type = (self.type or "mlp").lower()
        if proj_type != "mlp":
            raise ValueError(
                f"Unsupported projector type '{proj_type}'. Only 'mlp' is supported. "
                "Legacy LSTM projector checkpoints are no longer supported."
            )
        return proj_type


@dataclass
class PipelineConfig:
    data: DataConfig
    detector: DetectorConfig
    embedding: EmbeddingConfig
    seed: SeedConfig
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    use_stylegan_mapper: bool = False

    @staticmethod
    def _require(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise KeyError(f"Missing required config key: {key}")
        return mapping[key]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineConfig":
        data_cfg = payload.get("data", {})
        detector_cfg = cls._require(payload, "detector")
        embedding_cfg = cls._require(payload, "embedding")
        seed_cfg = cls._require(payload, "seed")
        projector_cfg = payload.get("projector", {})

        def _path_or_none(value: Optional[str]) -> Optional[Path]:
            return Path(value) if value else None

        feature_selector_cfg = FeatureSelectorConfig(**embedding_cfg.get("feature_selector", {}))
        embedding_kwargs = {
            **embedding_cfg,
            "method": embedding_cfg.get("method") or embedding_cfg.get("type") or "facenet",
            "pretrained": embedding_cfg.get("pretrained", "vggface2"),
            "device": embedding_cfg.get("device"),
            "feature_selector": feature_selector_cfg,
        }

        projector = ProjectorConfig(
            type=projector_cfg.get("type", projector_cfg.get("arch", "mlp")),
            key_dim=int(projector_cfg.get("key_dim", 128)),
            hidden_dims=tuple(projector_cfg.get("hidden_dims", (1024, 512))),
            dropout=float(projector_cfg.get("dropout", 0.0)),
            enable_input_l2_norm=bool(projector_cfg.get("enable_input_l2_norm", projector_cfg.get("output_l2_normalize", True))),
            enable_key_upscaler=bool(projector_cfg.get("enable_key_upscaler", True)),
        )

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
            seed=SeedConfig(**seed_cfg),
            projector=projector,
            use_stylegan_mapper=bool(payload.get("use_stylegan_mapper", False)),
        )
