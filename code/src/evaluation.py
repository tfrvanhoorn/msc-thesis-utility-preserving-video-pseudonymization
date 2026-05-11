from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[0]
PROJECT_ROOT = current_file.parents[1]
EXTERNAL_LIB_ROOT = PROJECT_ROOT / "external_libraries"

# Silence PyTorch bilinear align_corners warning
warnings.filterwarnings(
    "ignore",
    message="Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0",
    category=UserWarning,
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXTERNAL_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_LIB_ROOT))

from components import ArcFaceEmbedder, EmbeddingModel, FacenetEmbedder, MTCNNAligner, MTCNNDetector  # noqa: E402
from fid_metric import FidEvaluator  # noqa: E402
from landmark_metrics import LandmarkDistanceEvaluator  # noqa: E402
from metrics import MetricsAccumulator  # noqa: E402
from perceptual_metrics import PerceptualEvaluator  # noqa: E402
from data.prepared import (  # noqa: E402
    DEFAULT_PREPARED_REGEX,
    PreparedNameError,
    collect_prepared_videos,
    compile_prepared_regex,
    map_prepared_videos_by_key,
)
from data.video_io import load_video_frames  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


SUPPORTED_METRICS = {
    "detection",
    "anonymization",
    "synchronism",
    "diversity",
    "differentiation",
    "landmark_distance",
    "fid",
    "lpips",
    "ssim",
}

KEY_DIR_PATTERN = re.compile(r"^key(?P<index>\d+)$")


@dataclass(frozen=True)
class EvalEntry:
    identity: str
    youtube_id: str | None
    source_id: str
    input_video: Path
    outputs: dict[str, Path]


def _log_pipe(category: str, **fields: Any) -> None:
    parts = [category]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logging.info(" | ".join(parts))


def _to_uint8_rgb_image(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 3:
        raise ValueError(f"Expected CHW tensor image, got shape {tuple(image.shape)}")
    chw = image.detach().to("cpu")
    if chw.dtype != torch.float32:
        chw = chw.float()
    chw = chw.clamp(0.0, 1.0)
    hwc = chw.permute(1, 2, 0).numpy()
    return (hwc * 255.0).round().astype(np.uint8)


def _parse_metrics(value: str) -> set[str]:
    raw_items = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw_items:
        raise ValueError("--metrics must include at least one metric")
    if "all" in raw_items:
        return set(SUPPORTED_METRICS)
    unknown = sorted(set(raw_items) - SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"Unsupported metric names in --metrics: {', '.join(unknown)}")
    return set(raw_items)


def _sorted_key_names(keys: list[str]) -> list[str]:
    def _key_rank(name: str) -> tuple[int, str]:
        if name.startswith("key"):
            suffix = name[3:]
            if suffix.isdigit():
                return int(suffix), name
        return 10**9, name

    return sorted(keys, key=_key_rank)


def _mask_disabled_metrics(summary: dict[str, Any], enabled: set[str]) -> None:
    if "detection" not in enabled:
        summary["detection_rate"] = None
        summary["detection_confidence"] = None
        summary["counts"]["detected_generated"] = 0
        summary["counts"]["total_generated"] = 0

    if "anonymization" not in enabled:
        summary["anonymization"].update({"auc": None, "eer": None, "eer_threshold": None})
        summary["anonymization"]["counts"] = {"total": 0}
        summary["counts"]["anonymization_total"] = 0

    if "synchronism" not in enabled:
        for key in ("synchronism_total", "synchronism_within", "synchronism_cross"):
            summary[key].update({"auc": None, "eer": None, "eer_threshold": None})
            summary[key]["counts"] = {"total": 0}
        summary["counts"]["synchronism_total"] = 0
        summary["counts"]["synchronism_within_total"] = 0
        summary["counts"]["synchronism_cross_total"] = 0

    if "diversity" not in enabled:
        summary["diversity"].update({"auc": None, "eer": None, "eer_threshold": None})
        summary["diversity"]["counts"] = {"total": 0}
        summary["counts"]["diversity_total"] = 0

    if "differentiation" not in enabled:
        summary["differentiation"].update({"auc": None, "eer": None, "eer_threshold": None})
        summary["differentiation"]["counts"] = {"total": 0}
        summary["counts"]["differentiation_total"] = 0

    if "landmark_distance" not in enabled:
        summary["landmark_distance"] = None
        summary["landmark_utility"]["landmark_distance"] = None
        summary["landmark_utility"]["counts"] = {"valid_pairs": 0, "invalid_pairs": 0}
        summary["counts"]["landmark_pairs_valid"] = 0
        summary["counts"]["landmark_pairs_invalid"] = 0

    if "fid" not in enabled:
        summary["fid"] = None
        summary["realism_utility"]["fid"] = None
        summary["realism_utility"]["counts"] = {"real_frames": 0, "generated_frames": 0}
        summary["counts"]["fid_real_frames"] = 0
        summary["counts"]["fid_generated_frames"] = 0

    if "lpips" not in enabled:
        summary["lpips_distance"] = None
        summary["perceptual_utility"]["lpips_distance"] = None
        summary["perceptual_utility"]["counts"]["lpips_valid_pairs"] = 0
        summary["perceptual_utility"]["counts"]["lpips_invalid_pairs"] = 0
        summary["counts"]["lpips_pairs_valid"] = 0
        summary["counts"]["lpips_pairs_invalid"] = 0

    if "ssim" not in enabled:
        summary["ssim_similarity"] = None
        summary["perceptual_utility"]["ssim_similarity"] = None
        summary["perceptual_utility"]["counts"]["ssim_valid_pairs"] = 0
        summary["perceptual_utility"]["counts"]["ssim_invalid_pairs"] = 0
        summary["counts"]["ssim_pairs_valid"] = 0
        summary["counts"]["ssim_pairs_invalid"] = 0


def _build_chunk_score_snapshot(metrics: MetricsAccumulator, enabled: set[str]) -> dict[str, Any]:
    similar_pool = metrics._synchronism_total_hist.merged(  # type: ignore[attr-defined]
        [
            metrics._synchronism_total_hist,  # type: ignore[attr-defined]
            metrics._synchronism_within_hist,  # type: ignore[attr-defined]
            metrics._synchronism_cross_hist,  # type: ignore[attr-defined]
        ]
    )
    dissimilar_pool = metrics._anonymization_hist.merged(  # type: ignore[attr-defined]
        [
            metrics._anonymization_hist,  # type: ignore[attr-defined]
            metrics._diversity_hist,  # type: ignore[attr-defined]
            metrics._differentiation_hist,  # type: ignore[attr-defined]
        ]
    )

    def _auc_eer(
        positive_hist: Any,
        negative_hist: Any,
        *,
        positive_when_lower: bool,
        enabled_flag: bool,
    ) -> dict[str, float | None]:
        if not enabled_flag:
            return {"auc": None, "eer": None}
        values = MetricsAccumulator._compute_metric_auc_eer_from_hist(  # type: ignore[attr-defined]
            positive_hist,
            negative_hist,
            positive_when_lower=positive_when_lower,
        )
        return {"auc": values["auc"], "eer": values["eer"]}

    detection_rate = (
        float(metrics.detected_generated) / float(metrics.total_generated)
        if metrics.total_generated
        else 0.0
    )
    detection_confidence = (
        metrics.detection_score_sum / float(metrics.total_generated)
        if metrics.total_generated
        else 0.0
    )
    landmark_distance = (
        metrics.landmark_distance_sum / float(metrics.landmark_pairs_valid)
        if metrics.landmark_pairs_valid
        else None
    )
    lpips_distance = (
        metrics.lpips_distance_sum / float(metrics.lpips_pairs_valid)
        if metrics.lpips_pairs_valid
        else None
    )
    ssim_similarity = (
        metrics.ssim_similarity_sum / float(metrics.ssim_pairs_valid)
        if metrics.ssim_pairs_valid
        else None
    )

    return {
        "detection_rate": detection_rate if "detection" in enabled else None,
        "detection_confidence": detection_confidence if "detection" in enabled else None,
        "anonymization": _auc_eer(
            metrics._anonymization_hist,  # type: ignore[attr-defined]
            similar_pool,
            positive_when_lower=True,
            enabled_flag="anonymization" in enabled,
        ),
        "synchronism_total": _auc_eer(
            metrics._synchronism_total_hist,  # type: ignore[attr-defined]
            dissimilar_pool,
            positive_when_lower=False,
            enabled_flag="synchronism" in enabled,
        ),
        "synchronism_within": _auc_eer(
            metrics._synchronism_within_hist,  # type: ignore[attr-defined]
            dissimilar_pool,
            positive_when_lower=False,
            enabled_flag="synchronism" in enabled,
        ),
        "synchronism_cross": _auc_eer(
            metrics._synchronism_cross_hist,  # type: ignore[attr-defined]
            dissimilar_pool,
            positive_when_lower=False,
            enabled_flag="synchronism" in enabled,
        ),
        "diversity": _auc_eer(
            metrics._diversity_hist,  # type: ignore[attr-defined]
            similar_pool,
            positive_when_lower=True,
            enabled_flag="diversity" in enabled,
        ),
        "differentiation": _auc_eer(
            metrics._differentiation_hist,  # type: ignore[attr-defined]
            similar_pool,
            positive_when_lower=True,
            enabled_flag="differentiation" in enabled,
        ),
        "landmark_distance": landmark_distance if "landmark_distance" in enabled else None,
        "lpips_distance": lpips_distance if "lpips" in enabled else None,
        "ssim_similarity": ssim_similarity if "ssim" in enabled else None,
    }


def _fmt_metric_pair(metric_values: dict[str, float | None]) -> str:
    auc = metric_values.get("auc")
    eer = metric_values.get("eer")
    auc_text = "n/a" if auc is None else f"{auc:.4f}"
    eer_text = "n/a" if eer is None else f"{eer:.4f}"
    return f"auc={auc_text}, eer={eer_text}"


def _log_chunk_summary(
    *,
    chunk_idx: int,
    total_chunks: int,
    chunk_ids: list[str],
    scores: dict[str, Any],
    scope: str = "running",
) -> None:
    detection_rate = scores.get("detection_rate")
    detection_confidence = scores.get("detection_confidence")
    landmark_distance = scores.get("landmark_distance")
    lpips_distance = scores.get("lpips_distance")
    ssim_similarity = scores.get("ssim_similarity")
    detection_text = "n/a" if detection_rate is None else f"{float(detection_rate):.4f}"
    confidence_text = "n/a" if detection_confidence is None else f"{float(detection_confidence):.4f}"
    landmark_text = "n/a" if landmark_distance is None else f"{float(landmark_distance):.4f}"
    lpips_text = "n/a" if lpips_distance is None else f"{float(lpips_distance):.4f}"
    ssim_text = "n/a" if ssim_similarity is None else f"{float(ssim_similarity):.4f}"
    id_text = ", ".join(chunk_ids) if chunk_ids else "none"

    logging.info(
        "chunk %d/%d | scope=%s | ids=[%s]\n"
        "  detection_rate=%s | detection_confidence=%s | anonymization(%s) | synchronism_total(%s) | synchronism_within(%s) | synchronism_cross(%s)\n"
        "  diversity(%s) | differentiation(%s) | landmark_distance=%s | lpips=%s | ssim=%s",
        chunk_idx,
        total_chunks,
        scope,
        id_text,
        detection_text,
        confidence_text,
        _fmt_metric_pair(scores["anonymization"]),
        _fmt_metric_pair(scores["synchronism_total"]),
        _fmt_metric_pair(scores["synchronism_within"]),
        _fmt_metric_pair(scores["synchronism_cross"]),
        _fmt_metric_pair(scores["diversity"]),
        _fmt_metric_pair(scores["differentiation"]),
        landmark_text,
        lpips_text,
        ssim_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inferred videos from prepared input/inferred folders")

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing prepared input videos named {id}_sample{count}_{original_filename}.mp4",
    )
    parser.add_argument(
        "--inferred_dir",
        type=Path,
        required=True,
        help="Directory containing inferred videos. Use --inferred_nested_keys for key1/key2/... subfolders.",
    )
    parser.add_argument(
        "--inferred_nested_keys",
        action="store_true",
        help="Set when inferred_dir contains nested key folders named key1, key2, ...",
    )
    parser.add_argument(
        "--filename_regex",
        type=str,
        default=DEFAULT_PREPARED_REGEX,
        help=(
            "Regex used to parse prepared filenames; must define named groups "
            "identity, sample, original"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SRC_ROOT / "eval_results",
        help="Directory to save evaluation reports",
    )

    parser.add_argument("--num_keys", type=int, default=None, help="Required key count; defaults to manifest value or inferred from files")
    parser.add_argument("--detection_key", type=int, default=1, help="Key index used for detection/anonymization/synchronism/differentiation branches")

    parser.add_argument("--max_identities", type=int, default=None, help="Optional cap on number of identities to evaluate")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Optional cap on number of videos per identity to evaluate")
    parser.add_argument(
        "--ids_per_chunk",
        type=int,
        default=2,
        help="Identities per chunk for chunked evaluation and differentiation aggregation",
    )
    parser.add_argument(
        "--samples_per_id_per_chunk",
        type=int,
        default=2,
        help="Sample videos per identity per chunk and embedding chunk size for synchronism/differentiation",
    )
    parser.add_argument(
        "--keys_per_id_per_chunk",
        type=int,
        default=2,
        help="Keys per identity per chunk (applies only when --inferred_nested_keys is set)",
    )
    parser.add_argument(
        "--landmark_shape_predictor",
        type=Path,
        default=None,
        help="Path to dlib shape predictor model file (e.g., shape_predictor_68_face_landmarks.dat)",
    )
    parser.add_argument(
        "--landmark_detector_upsample",
        type=int,
        default=0,
        help="Number of image upsample steps for dlib frontal face detector",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="detection,anonymization,synchronism,diversity,differentiation,landmark_distance,lpips,ssim",
        help=(
            "Comma-separated metrics list (supported: detection, anonymization, synchronism, diversity, "
            "differentiation, landmark_distance, fid, lpips, ssim; use 'all' for full set)"
        ),
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone network when LPIPS metric is enabled",
    )
    parser.add_argument(
        "--lpips_cache_dir",
        type=Path,
        default=None,
        help="Optional cache directory for LPIPS/Torch model weights (TORCH_HOME)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=["facenet", "arcface"],
        default="facenet",
        help="Embedding backend used for identity metrics",
    )
    parser.add_argument(
        "--arcface_model_name",
        type=str,
        default="buffalo_l",
        help="InsightFace ArcFace model pack name",
    )
    parser.add_argument(
        "--arcface_cache_dir",
        type=Path,
        default=None,
        help="Optional ArcFace cache directory (sets INSIGHTFACE_HOME)",
    )
    parser.add_argument(
        "--arcface_auto_download",
        dest="arcface_auto_download",
        action="store_true",
        help="Allow ArcFace model download when not present in cache",
    )
    parser.add_argument(
        "--no_arcface_auto_download",
        dest="arcface_auto_download",
        action="store_false",
        help="Disable ArcFace model download and require a pre-populated cache",
    )
    parser.set_defaults(arcface_auto_download=True)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    return parser.parse_args()


def _load_video_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = load_video_frames(path, max_frames=None, frame_step=1, convert_rgb=True)
    if arr is None:
        return torch.empty((0, 3, 0, 0), device=device)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device) / 255.0


def _collect_detected_faces(
    detector: MTCNNDetector,
    aligner: MTCNNAligner,
    embedder: EmbeddingModel,
    sample_frames: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aligned_faces: list[torch.Tensor] = []
    input_mask_list: list[bool] = []
    score_list: list[float] = []

    if sample_frames.numel() == 0:
        emb_dim = int(getattr(embedder, "embedding_size", 512))
        return (
            torch.empty((0, emb_dim), device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    for frame in sample_frames:
        detections = detector.detect(frame)
        if detections:
            top = max(detections, key=lambda d: d.score)
            aligned = aligner.align(frame, top).to(device)
            aligned_faces.append(aligned)
            input_mask_list.append(True)
            score_list.append(float(top.score))
        else:
            aligned_faces.append(torch.empty(0, device=device))
            input_mask_list.append(False)
            score_list.append(0.0)

    input_mask = torch.tensor(input_mask_list, dtype=torch.bool, device=device)
    scores = torch.tensor(score_list, dtype=torch.float32, device=device)
    if input_mask.any():
        valid_idx = [i for i, is_valid in enumerate(input_mask_list) if is_valid]
        valid_faces = [aligned_faces[i] for i in valid_idx]
        embeds = embedder.embed(valid_faces, with_grad=False)
        embed_dim = int(embeds.shape[1])
        full_embeds = torch.zeros((len(aligned_faces), embed_dim), device=device, dtype=embeds.dtype)
        for emb, idx in zip(embeds, valid_idx):
            full_embeds[idx] = emb
    else:
        emb_dim = int(getattr(embedder, "embedding_size", 512))
        full_embeds = torch.zeros((len(aligned_faces), emb_dim), device=device, dtype=torch.float32)
    return full_embeds, input_mask, scores


def _sorted_sample_keys(keys: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(keys, key=lambda item: (item[0], item[1]))


def _discover_key_video_maps(
    inferred_dir: Path,
    *,
    nested_keys: bool,
    filename_regex: re.Pattern[str],
) -> dict[str, dict[tuple[str, int], Path]]:
    if nested_keys:
        key_maps: dict[str, dict[tuple[str, int], Path]] = {}
        for child in sorted([p for p in inferred_dir.iterdir() if p.is_dir()]):
            match = KEY_DIR_PATTERN.match(child.name)
            if match is None:
                continue
            key_name = f"key{int(match.group('index'))}"
            refs = collect_prepared_videos(child, filename_regex)
            key_maps[key_name] = {ref.key: ref.video_path for ref in refs}
        if not key_maps:
            raise ValueError(
                f"No key folders found in inferred_dir={inferred_dir}. Expected folders named key1, key2, ..."
            )
        return key_maps

    refs = collect_prepared_videos(inferred_dir, filename_regex)
    return {"key1": {ref.key: ref.video_path for ref in refs}}


def _load_entries_from_prepared_dirs(
    input_dir: Path,
    inferred_dir: Path,
    *,
    nested_keys: bool,
    filename_regex_pattern: str,
    required_num_keys: int | None,
) -> tuple[list[EvalEntry], int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not inferred_dir.exists():
        raise FileNotFoundError(f"Inferred directory not found: {inferred_dir}")

    filename_regex = compile_prepared_regex(filename_regex_pattern)
    input_refs = collect_prepared_videos(input_dir, filename_regex)
    if not input_refs:
        raise FileNotFoundError(f"No prepared input videos found in {input_dir}")

    input_map = map_prepared_videos_by_key(input_refs)
    key_video_maps = _discover_key_video_maps(
        inferred_dir,
        nested_keys=nested_keys,
        filename_regex=filename_regex,
    )

    discovered_key_names = _sorted_key_names(list(key_video_maps.keys()))
    if required_num_keys is not None:
        if required_num_keys <= 0:
            raise ValueError("--num_keys must be >= 1 when provided")
        key_names = [f"key{i}" for i in range(1, required_num_keys + 1)]
    else:
        key_names = discovered_key_names

    missing_key_dirs = [key_name for key_name in key_names if key_name not in key_video_maps]
    if missing_key_dirs:
        raise FileNotFoundError(
            "Missing key folders or outputs in inferred directory: "
            + ", ".join(missing_key_dirs)
        )

    entries: list[EvalEntry] = []
    missing_pairs: list[str] = []

    for sample_key in _sorted_sample_keys(list(input_map.keys())):
        input_ref = input_map[sample_key]
        outputs: dict[str, Path] = {}
        for key_name in key_names:
            output_path = key_video_maps[key_name].get(sample_key)
            if output_path is None:
                missing_pairs.append(
                    f"identity={sample_key[0]} sample={sample_key[1]} key={key_name}"
                )
                continue
            outputs[key_name] = output_path

        source_id = f"{input_ref.identity}_sample{input_ref.sample_index}_{input_ref.original_name}"
        entries.append(
            EvalEntry(
                identity=input_ref.identity,
                youtube_id=None,
                source_id=source_id,
                input_video=input_ref.video_path,
                outputs=outputs,
            )
        )

    if missing_pairs:
        preview = "; ".join(missing_pairs[:20])
        if len(missing_pairs) > 20:
            preview += f"; ... ({len(missing_pairs)} total missing pairs)"
        raise FileNotFoundError(
            "Missing inferred outputs for prepared inputs. "
            f"Examples: {preview}"
        )

    return entries, len(key_names)


def _apply_entry_caps(
    entries: list[EvalEntry],
    *,
    max_identities: int | None,
    max_videos_per_identity: int | None,
) -> list[EvalEntry]:
    if max_identities is None and max_videos_per_identity is None:
        return entries

    kept: list[EvalEntry] = []
    seen_identities: set[str] = set()
    per_identity_count: dict[str, int] = {}

    for entry in entries:
        identity = entry.identity
        if identity not in seen_identities:
            if max_identities is not None and len(seen_identities) >= max_identities:
                continue
            seen_identities.add(identity)

        if max_videos_per_identity is not None:
            current = per_identity_count.get(identity, 0)
            if current >= max_videos_per_identity:
                continue
            per_identity_count[identity] = current + 1

        kept.append(entry)

    return kept


def _build_entry_chunks(
    entries: list[EvalEntry],
    *,
    ids_per_chunk: int,
    samples_per_id_per_chunk: int,
) -> list[tuple[list[EvalEntry], bool]]:
    by_identity: dict[str, list[EvalEntry]] = {}
    for entry in entries:
        by_identity.setdefault(entry.identity, []).append(entry)

    identity_order = sorted(by_identity.keys())
    identity_queues: dict[str, list[EvalEntry]] = {identity: list(by_identity[identity]) for identity in identity_order}
    chunks: list[tuple[list[EvalEntry], bool]] = []

    while True:
        eligible_ids = [identity for identity in identity_order if len(identity_queues[identity]) >= samples_per_id_per_chunk]
        if len(eligible_ids) < ids_per_chunk:
            break

        selected_ids = eligible_ids[:ids_per_chunk]
        chunk_entries: list[EvalEntry] = []
        for identity in selected_ids:
            take = identity_queues[identity][:samples_per_id_per_chunk]
            identity_queues[identity] = identity_queues[identity][samples_per_id_per_chunk:]
            chunk_entries.extend(take)
        chunks.append((chunk_entries, True))

    leftovers: list[EvalEntry] = []
    for identity in identity_order:
        leftovers.extend(identity_queues[identity])

    if leftovers:
        fallback_size = max(ids_per_chunk * samples_per_id_per_chunk, 1)
        for start in range(0, len(leftovers), fallback_size):
            chunks.append((leftovers[start : start + fallback_size], False))

    return chunks


def _build_face_embedder(args: argparse.Namespace, device: torch.device) -> EmbeddingModel:
    if args.embedding_model == "arcface":
        return ArcFaceEmbedder(
            model_name=args.arcface_model_name,
            device=str(device),
            cache_dir=args.arcface_cache_dir,
            auto_download=bool(args.arcface_auto_download),
        )
    return FacenetEmbedder(pretrained="vggface2", device=str(device))


def main() -> None:
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)
    enabled_metrics = _parse_metrics(args.metrics)
    detection_branch_enabled = bool(enabled_metrics & {"detection", "anonymization", "synchronism", "differentiation"})
    diversity_enabled = "diversity" in enabled_metrics
    landmark_distance_enabled = "landmark_distance" in enabled_metrics
    fid_enabled = "fid" in enabled_metrics
    lpips_enabled = "lpips" in enabled_metrics
    ssim_enabled = "ssim" in enabled_metrics
    perceptual_enabled = lpips_enabled or ssim_enabled
    realism_enabled = perceptual_enabled or fid_enabled

    if args.max_identities is not None and args.max_identities <= 0:
        raise ValueError("--max_identities must be > 0 when provided")
    if args.max_videos_per_identity is not None and args.max_videos_per_identity <= 0:
        raise ValueError("--max_videos_per_identity must be > 0 when provided")
    if args.landmark_detector_upsample < 0:
        raise ValueError("--landmark_detector_upsample must be >= 0")

    ids_per_chunk = int(args.ids_per_chunk)
    samples_per_id_per_chunk = int(args.samples_per_id_per_chunk)
    keys_per_id_per_chunk = int(args.keys_per_id_per_chunk)

    if ids_per_chunk < 2:
        raise ValueError("--ids_per_chunk must be >= 2")
    if samples_per_id_per_chunk < 2:
        raise ValueError("--samples_per_id_per_chunk must be >= 2")
    if keys_per_id_per_chunk < 2:
        raise ValueError("--keys_per_id_per_chunk must be >= 2")
    if landmark_distance_enabled and args.landmark_shape_predictor is None:
        raise ValueError("--landmark_shape_predictor is required when landmark_distance metric is enabled")

    try:
        entries, num_keys = _load_entries_from_prepared_dirs(
            args.input_dir,
            args.inferred_dir,
            nested_keys=args.inferred_nested_keys,
            filename_regex_pattern=args.filename_regex,
            required_num_keys=args.num_keys,
        )
    except PreparedNameError as exc:
        raise ValueError(str(exc)) from exc

    if num_keys < 1:
        raise ValueError("--num_keys must be >= 1")
    if "diversity" in enabled_metrics and not args.inferred_nested_keys:
        raise ValueError("Diversity metric requires --inferred_nested_keys and at least two key folders")
    if "diversity" in enabled_metrics and num_keys < 2:
        raise ValueError("Diversity metric requires at least two keys")

    entries = _apply_entry_caps(
        entries,
        max_identities=args.max_identities,
        max_videos_per_identity=args.max_videos_per_identity,
    )
    entry_chunks = _build_entry_chunks(
        entries,
        ids_per_chunk=ids_per_chunk,
        samples_per_id_per_chunk=samples_per_id_per_chunk,
    )

    key_names = [f"key{i}" for i in range(1, num_keys + 1)]
    required_detection_key = f"key{args.detection_key}"
    perceptual_key_name = required_detection_key if args.inferred_nested_keys else "key1"
    if detection_branch_enabled and required_detection_key not in key_names:
        raise ValueError(f"--detection_key must be in [1, {num_keys}]")
    if realism_enabled and perceptual_key_name not in key_names:
        raise ValueError(f"Perceptual metric key must be available in [1, {num_keys}]")

    _log_pipe(
        "evaluation_start",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        num_entries=len(entries),
        num_chunks=len(entry_chunks),
        num_keys=num_keys,
        detection_key=required_detection_key,
        enabled_metrics=",".join(sorted(enabled_metrics)),
        lpips_net=args.lpips_net,
        lpips_cache_dir=str(args.lpips_cache_dir) if args.lpips_cache_dir is not None else None,
        embedding_model=args.embedding_model,
        arcface_model_name=args.arcface_model_name,
        arcface_cache_dir=str(args.arcface_cache_dir) if args.arcface_cache_dir is not None else None,
        arcface_auto_download=bool(args.arcface_auto_download),
        ids_per_chunk=ids_per_chunk,
        samples_per_id_per_chunk=samples_per_id_per_chunk,
        keys_per_id_per_chunk=keys_per_id_per_chunk,
        landmark_shape_predictor=str(args.landmark_shape_predictor) if args.landmark_shape_predictor is not None else None,
        landmark_detector_upsample=int(args.landmark_detector_upsample),
    )

    detector = MTCNNDetector(
        image_size=256,
        margin=0,
        score_threshold=0.3,
        min_face_size=12,
        max_faces=None,
        keep_all=True,
        post_process=False,
        device=str(device),
    )
    aligner = MTCNNAligner(output_size=256)
    embedder = _build_face_embedder(args, device)

    metrics = MetricsAccumulator(
        synchronism_chunk_size=samples_per_id_per_chunk,
        show_progress=False,
        anonymization_enabled="anonymization" in enabled_metrics,
        diversity_enabled=diversity_enabled,
    )
    landmark_evaluator = (
        LandmarkDistanceEvaluator(
            shape_predictor_path=args.landmark_shape_predictor,
            detector_upsample=args.landmark_detector_upsample,
        )
        if landmark_distance_enabled
        else None
    )
    perceptual_evaluator = (
        PerceptualEvaluator(
            device=device,
            compute_lpips=lpips_enabled,
            compute_ssim=ssim_enabled,
            lpips_net=args.lpips_net,
            lpips_cache_dir=args.lpips_cache_dir,
        )
        if perceptual_enabled
        else None
    )
    fid_evaluator = FidEvaluator(device=device) if fid_enabled else None

    identity_to_label: dict[str, int] = {}
    batch_processing_start_time = time.perf_counter()
    total_samples = 0

    try:
        with torch.no_grad():
            total_entries = len(entries)
            progress = tqdm(
                total=total_entries,
                desc="Evaluating",
                unit="entry",
                dynamic_ncols=True,
                smoothing=0.1,
            )
            processed_entries = 0
            for chunk_idx, (chunk_entries, chunk_eligible) in enumerate(entry_chunks, start=1):
                chunk_diff_embeddings_by_label: dict[int, list[torch.Tensor]] = {}
                chunk_metrics = MetricsAccumulator(
                    synchronism_chunk_size=samples_per_id_per_chunk,
                    show_progress=False,
                    anonymization_enabled="anonymization" in enabled_metrics,
                    diversity_enabled=diversity_enabled,
                )
                chunk_diff_embeddings_by_label_local: dict[int, list[torch.Tensor]] = {}
                chunk_entry_total = len(chunk_entries)
                chunk_entry_done = 0

                for entry in chunk_entries:
                    progress.update(1)
                    processed_entries += 1
                    chunk_entry_done += 1
                    if entry.identity not in identity_to_label:
                        identity_to_label[entry.identity] = len(identity_to_label)
                    label = identity_to_label[entry.identity]

                    available_key_names = [k for k in key_names if k in entry.outputs]
                    if detection_branch_enabled and required_detection_key not in available_key_names:
                        raise FileNotFoundError(
                            f"Entry {entry.input_video} is missing required detection branch output: {required_detection_key}"
                        )
                    if perceptual_enabled and perceptual_key_name not in available_key_names:
                        raise FileNotFoundError(
                            f"Entry {entry.input_video} is missing required perceptual branch output: {perceptual_key_name}"
                        )

                    input_frames = _load_video_tensor(entry.input_video, device)
                    if input_frames.shape[0] == 0:
                        continue

                    real_embeddings, input_mask, _ = _collect_detected_faces(
                        detector,
                        aligner,
                        embedder,
                        input_frames,
                        device,
                    )

                    base_real_embeddings = real_embeddings
                    base_input_mask = input_mask
                    key_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                    sorted_key_names = _sorted_key_names(available_key_names)
                    if args.inferred_nested_keys:
                        selected: list[str] = []
                        for required_key in (required_detection_key, perceptual_key_name):
                            if required_key in sorted_key_names and required_key not in selected:
                                selected.append(required_key)
                        for key_name in sorted_key_names:
                            if key_name not in selected:
                                selected.append(key_name)
                            if len(selected) >= keys_per_id_per_chunk:
                                break
                        eval_key_names = selected
                    else:
                        eval_key_names = sorted_key_names

                    if detection_branch_enabled and required_detection_key not in eval_key_names:
                        eval_key_names = [required_detection_key] + [k for k in eval_key_names if k != required_detection_key]
                    if diversity_enabled and len(eval_key_names) < 2:
                        raise ValueError(f"Diversity is enabled but entry {entry.input_video} has fewer than 2 selected outputs")

                    progress.set_postfix(
                        {
                            "chunk": f"{chunk_idx}/{len(entry_chunks)}",
                            "chunk_pos": f"{chunk_entry_done}/{chunk_entry_total}",
                            "eligible": "Y" if chunk_eligible else "N",
                            "id": entry.identity,
                            "keys": len(eval_key_names),
                            "seen": processed_entries,
                        },
                        refresh=False,
                    )

                    for key_name in eval_key_names:
                        output_path = entry.outputs[key_name]
                        output_frames = _load_video_tensor(output_path, device)
                        if output_frames.shape[0] == 0:
                            empty_real = base_real_embeddings[:0]
                            empty_mask = base_input_mask[:0]
                            virt = torch.zeros_like(empty_real)
                            gmask = torch.zeros((0,), dtype=torch.bool, device=device)
                            scores = torch.zeros((0,), dtype=torch.float32, device=device)
                            key_results[key_name] = (empty_real, empty_mask, virt, gmask, scores)
                            continue

                        raw_output_frames = output_frames
                        min_len = min(int(base_real_embeddings.shape[0]), int(output_frames.shape[0]))
                        key_real_embeddings = base_real_embeddings[:min_len]
                        key_input_mask = base_input_mask[:min_len]
                        key_input_frames = input_frames[:min_len]
                        output_frames = output_frames[:min_len]

                        if fid_enabled and key_name == perceptual_key_name and fid_evaluator is not None:
                            fid_evaluator.update_real(input_frames)
                            fid_evaluator.update_generated(raw_output_frames)

                        if perceptual_enabled and key_name == perceptual_key_name and perceptual_evaluator is not None:
                            aligned_inputs, aligned_outputs = perceptual_evaluator.prepare_video_pair(key_input_frames, output_frames)
                            pair_count = int(aligned_inputs.shape[0])
                            for frame_idx in range(pair_count):
                                lpips_value, ssim_value = perceptual_evaluator.compute_frame_pair(
                                    aligned_inputs[frame_idx],
                                    aligned_outputs[frame_idx],
                                )
                                metrics.update_perceptual_utility(lpips_value, ssim_value)
                                chunk_metrics.update_perceptual_utility(lpips_value, ssim_value)

                        virt_embeddings, gen_mask, gen_scores = _collect_detected_faces(
                            detector,
                            aligner,
                            embedder,
                            output_frames,
                            device,
                        )
                        key_results[key_name] = (
                            key_real_embeddings,
                            key_input_mask,
                            virt_embeddings,
                            gen_mask,
                            gen_scores,
                        )

                        if landmark_distance_enabled and landmark_evaluator is not None:
                            pair_count = min(int(input_frames.shape[0]), int(output_frames.shape[0]))
                            for frame_idx in range(pair_count):
                                in_img = _to_uint8_rgb_image(input_frames[frame_idx])
                                out_img = _to_uint8_rgb_image(output_frames[frame_idx])
                                dist = landmark_evaluator.compute_pair_distance(in_img, out_img)
                                metrics.update_landmark_distance(dist.distance)
                                chunk_metrics.update_landmark_distance(dist.distance)

                        del output_frames

                    det_real_embeddings = torch.empty((0, 0), device=device)
                    det_input_mask = torch.empty((0,), dtype=torch.bool, device=device)
                    det_embeddings = torch.empty((0, 0), device=device)
                    det_gen_mask = torch.empty((0,), dtype=torch.bool, device=device)
                    det_scores = torch.empty((0,), dtype=torch.float32, device=device)
                    valid_mask = torch.empty((0,), dtype=torch.bool, device=device)
                    if detection_branch_enabled:
                        det_real_embeddings, det_input_mask, det_embeddings, det_gen_mask, det_scores = key_results[required_detection_key]
                        valid_mask = det_input_mask & det_gen_mask

                    if "detection" in enabled_metrics:
                        metrics.update_detection(det_gen_mask, det_scores)
                        chunk_metrics.update_detection(det_gen_mask, det_scores)
                    if "anonymization" in enabled_metrics:
                        metrics.update_anonymization(det_real_embeddings, det_embeddings, valid_mask)
                        chunk_metrics.update_anonymization(det_real_embeddings, det_embeddings, valid_mask)

                    valid_virtual = det_embeddings[valid_mask] if detection_branch_enabled else torch.empty((0, 0), device=device)
                    if chunk_eligible and "synchronism" in enabled_metrics and valid_virtual.numel() > 0:
                        metrics.add_synchronism_embeddings(label, valid_virtual, source_id=entry.source_id)
                        chunk_metrics.add_synchronism_embeddings(label, valid_virtual, source_id=entry.source_id)
                    if chunk_eligible and "differentiation" in enabled_metrics and valid_virtual.numel() > 0:
                        chunk_diff_embeddings_by_label.setdefault(label, []).append(valid_virtual.detach().to("cpu"))
                        chunk_diff_embeddings_by_label_local.setdefault(label, []).append(valid_virtual.detach().to("cpu"))

                    for key_a, key_b in itertools.combinations(eval_key_names, 2):
                        _, input_mask_a, emb_a, mask_a, _ = key_results[key_a]
                        _, input_mask_b, emb_b, mask_b, _ = key_results[key_b]
                        pair_len = min(
                            int(input_mask_a.shape[0]),
                            int(input_mask_b.shape[0]),
                            int(mask_a.shape[0]),
                            int(mask_b.shape[0]),
                            int(emb_a.shape[0]),
                            int(emb_b.shape[0]),
                        )
                        if pair_len <= 0:
                            continue
                        pair_mask = (
                            input_mask_a[:pair_len]
                            & input_mask_b[:pair_len]
                            & mask_a[:pair_len]
                            & mask_b[:pair_len]
                        )
                        if diversity_enabled and pair_mask.any():
                            pair_emb_a = emb_a[:pair_len][pair_mask].detach().to("cpu")
                            pair_emb_b = emb_b[:pair_len][pair_mask].detach().to("cpu")
                            metrics.update_diversity(
                                pair_emb_a,
                                pair_emb_b,
                                embedding_chunk_size=samples_per_id_per_chunk,
                            )
                            chunk_metrics.update_diversity(
                                pair_emb_a,
                                pair_emb_b,
                                embedding_chunk_size=samples_per_id_per_chunk,
                            )

                    del key_results
                    del input_frames
                    del base_real_embeddings

                    total_samples += 1

                    valid_frames = int(valid_mask.sum().item()) if detection_branch_enabled else 0
                    progress.set_postfix(
                        {
                            "chunk": f"{chunk_idx}/{len(entry_chunks)}",
                            "chunk_pos": f"{chunk_entry_done}/{chunk_entry_total}",
                            "eligible": "Y" if chunk_eligible else "N",
                            "id": entry.identity,
                            "keys": len(eval_key_names),
                            "valid": valid_frames,
                            "samples": total_samples,
                        },
                        refresh=False,
                    )

                if chunk_eligible and "differentiation" in enabled_metrics and len(chunk_diff_embeddings_by_label) >= 2:
                    metrics.update_differentiation_batched(
                        chunk_diff_embeddings_by_label,
                        identity_block_size=ids_per_chunk,
                        embedding_chunk_size=samples_per_id_per_chunk,
                        show_progress=False,
                    )
                if chunk_eligible and "differentiation" in enabled_metrics and len(chunk_diff_embeddings_by_label_local) >= 2:
                    chunk_metrics.update_differentiation_batched(
                        chunk_diff_embeddings_by_label_local,
                        identity_block_size=ids_per_chunk,
                        embedding_chunk_size=samples_per_id_per_chunk,
                        show_progress=False,
                    )

                if chunk_eligible and "synchronism" in enabled_metrics:
                    metrics.flush_synchronism_chunk()
                    chunk_metrics.flush_synchronism_chunk()

                chunk_ids = sorted({entry.identity for entry in chunk_entries})
                chunk_scores = _build_chunk_score_snapshot(metrics, enabled_metrics)
                chunk_only_scores = _build_chunk_score_snapshot(chunk_metrics, enabled_metrics)
                _log_chunk_summary(
                    chunk_idx=chunk_idx,
                    total_chunks=len(entry_chunks),
                    chunk_ids=chunk_ids,
                    scores=chunk_only_scores,
                    scope="chunk_only",
                )
                _log_chunk_summary(
                    chunk_idx=chunk_idx,
                    total_chunks=len(entry_chunks),
                    chunk_ids=chunk_ids,
                    scores=chunk_scores,
                    scope="running",
                )

            progress.close()

    finally:
        if landmark_evaluator is not None:
            landmark_evaluator.close()

    batch_processing_end_time = time.perf_counter()
    batch_processing_seconds = max(0.0, batch_processing_end_time - batch_processing_start_time)

    summary = metrics.finalize()
    fid_value = fid_evaluator.compute() if fid_evaluator is not None else None
    summary["fid"] = fid_value
    summary["realism_utility"] = {
        "fid": fid_value,
        "counts": {
            "real_frames": int(fid_evaluator.real_frames) if fid_evaluator is not None else 0,
            "generated_frames": int(fid_evaluator.generated_frames) if fid_evaluator is not None else 0,
        },
    }
    summary["counts"]["fid_real_frames"] = int(fid_evaluator.real_frames) if fid_evaluator is not None else 0
    summary["counts"]["fid_generated_frames"] = int(fid_evaluator.generated_frames) if fid_evaluator is not None else 0
    _log_pipe("finalize_end")
    _mask_disabled_metrics(summary, enabled_metrics)
    _log_pipe(
        "evaluation_summary",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        total_samples=total_samples,
        batch_processing_seconds=batch_processing_seconds,
        detection_rate=summary["detection_rate"],
        detection_confidence=summary["detection_confidence"],
    )
    _log_pipe(
        "metric_anonymization",
        auc=summary["anonymization"]["auc"],
        eer=summary["anonymization"]["eer"],
        eer_threshold=summary["anonymization"]["eer_threshold"],
        total=summary["anonymization"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_total",
        auc=summary["synchronism_total"]["auc"],
        eer=summary["synchronism_total"]["eer"],
        eer_threshold=summary["synchronism_total"]["eer_threshold"],
        total=summary["synchronism_total"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_within",
        auc=summary["synchronism_within"]["auc"],
        eer=summary["synchronism_within"]["eer"],
        eer_threshold=summary["synchronism_within"]["eer_threshold"],
        total=summary["synchronism_within"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_cross",
        auc=summary["synchronism_cross"]["auc"],
        eer=summary["synchronism_cross"]["eer"],
        eer_threshold=summary["synchronism_cross"]["eer_threshold"],
        total=summary["synchronism_cross"]["counts"]["total"],
    )
    _log_pipe(
        "metric_diversity",
        auc=summary["diversity"]["auc"],
        eer=summary["diversity"]["eer"],
        eer_threshold=summary["diversity"]["eer_threshold"],
        total=summary["diversity"]["counts"]["total"],
    )
    _log_pipe(
        "metric_differentiation",
        auc=summary["differentiation"]["auc"],
        eer=summary["differentiation"]["eer"],
        eer_threshold=summary["differentiation"]["eer_threshold"],
        total=summary["differentiation"]["counts"]["total"],
    )
    _log_pipe(
        "metric_landmark_distance",
        distance=summary["landmark_distance"],
        valid_pairs=summary["landmark_utility"]["counts"]["valid_pairs"],
        invalid_pairs=summary["landmark_utility"]["counts"]["invalid_pairs"],
    )
    if fid_enabled:
        _log_pipe(
            "metric_fid",
            fid=summary["fid"],
            real_frames=summary["realism_utility"]["counts"]["real_frames"],
            generated_frames=summary["realism_utility"]["counts"]["generated_frames"],
        )
    _log_pipe(
        "metric_lpips",
        distance=summary["lpips_distance"],
        valid_pairs=summary["perceptual_utility"]["counts"]["lpips_valid_pairs"],
        invalid_pairs=summary["perceptual_utility"]["counts"]["lpips_invalid_pairs"],
    )
    _log_pipe(
        "metric_ssim",
        similarity=summary["ssim_similarity"],
        valid_pairs=summary["perceptual_utility"]["counts"]["ssim_valid_pairs"],
        invalid_pairs=summary["perceptual_utility"]["counts"]["ssim_invalid_pairs"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "folder_eval_report.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "inferred_dir": str(args.inferred_dir),
                "inferred_nested_keys": bool(args.inferred_nested_keys),
                "num_keys": num_keys,
                "enabled_metrics": sorted(enabled_metrics),
                "metrics": summary,
                "total_samples": total_samples,
                "timing": {
                    "batch_processing_seconds": batch_processing_seconds,
                },
                "settings": serialized_args,
                "identities": sorted(identity_to_label.keys()),
            },
            f,
            indent=2,
        )
    _log_pipe(
        "evaluation_report_saved",
        path=str(report_path),
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        batch_processing_seconds=batch_processing_seconds,
    )


if __name__ == "__main__":
    main()
