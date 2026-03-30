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

from components import FacenetEmbedder, MTCNNAligner, MTCNNDetector  # noqa: E402
from geometric_metrics import GeometricUtilityEvaluator  # noqa: E402
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
    "geometric",
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
        summary["counts"]["detected_generated"] = 0
        summary["counts"]["total_generated"] = 0

    if "anonymization" not in enabled:
        summary["anonymization_success_rate"] = None
        summary["anonymization"].update({"success_rate": None, "auc": None, "eer": None, "eer_threshold": None})
        summary["anonymization"]["counts"] = {"success": 0, "total": 0}
        summary["counts"]["anonymization_success"] = 0
        summary["counts"]["anonymization_total"] = 0

    if "synchronism" not in enabled:
        summary["synchronism_success_rate"] = None
        summary["synchronism_within_success_rate"] = None
        summary["synchronism_cross_success_rate"] = None
        for key in ("synchronism_total", "synchronism_within", "synchronism_cross"):
            summary[key].update({"success_rate": None, "auc": None, "eer": None, "eer_threshold": None})
            summary[key]["counts"] = {"success": 0, "total": 0}
        summary["counts"]["synchronism_success"] = 0
        summary["counts"]["synchronism_total"] = 0
        summary["counts"]["synchronism_within_success"] = 0
        summary["counts"]["synchronism_within_total"] = 0
        summary["counts"]["synchronism_cross_success"] = 0
        summary["counts"]["synchronism_cross_total"] = 0

    if "diversity" not in enabled:
        summary["diversity_success_rate"] = None
        summary["diversity"].update({"success_rate": None, "auc": None, "eer": None, "eer_threshold": None})
        summary["diversity"]["counts"] = {"success": 0, "total": 0}
        summary["counts"]["diversity_success"] = 0
        summary["counts"]["diversity_total"] = 0

    if "differentiation" not in enabled:
        summary["differentiation_success_rate"] = None
        summary["differentiation"].update({"success_rate": None, "auc": None, "eer": None, "eer_threshold": None})
        summary["differentiation"]["counts"] = {"success": 0, "total": 0}
        summary["counts"]["differentiation_success"] = 0
        summary["counts"]["differentiation_total"] = 0

    if "geometric" not in enabled:
        summary["head_posture_error"] = None
        summary["facial_expression_error"] = None
        summary["geometric_utility"]["head_posture_error"] = None
        summary["geometric_utility"]["facial_expression_error"] = None
        summary["geometric_utility"]["counts"] = {"valid_pairs": 0, "invalid_pairs": 0}
        summary["counts"]["geometric_pairs_valid"] = 0
        summary["counts"]["geometric_pairs_invalid"] = 0

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

    # Evaluation thresholds
    parser.add_argument("--ano_threshold", type=float, default=0.7, help="Cosine similarity threshold for anonymization success")
    parser.add_argument("--syn_threshold", type=float, default=0.7, help="Cosine similarity threshold for synchronism success")
    parser.add_argument(
        "--div_threshold",
        dest="div_threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for diversity success (same identity, different keys)",
    )
    parser.add_argument(
        "--diff_threshold",
        dest="diff_threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for differentiation success (different identities, same key)",
    )
    parser.add_argument(
        "--compute_auc_eer",
        action="store_true",
        help="Compute AUC/EER for anonymization, synchronism(all), diversity, and differentiation",
    )

    parser.add_argument("--max_identities", type=int, default=None, help="Optional cap on number of identities to evaluate")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Optional cap on number of videos per identity to evaluate")
    parser.add_argument(
        "--identities_per_batch",
        type=int,
        default=4,
        help="Identity block size used when aggregating differentiation at the end",
    )
    parser.add_argument(
        "--keys_per_batch",
        type=int,
        default=256,
        help="Embedding chunk size used for diversity/differentiation aggregation",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="detection,anonymization,synchronism,diversity,differentiation,geometric,lpips,ssim",
        help=(
            "Comma-separated metrics list (supported: detection, anonymization, synchronism, diversity, "
            "differentiation, geometric, lpips, ssim; use 'all' for full set)"
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
    embedder: FacenetEmbedder,
    sample_frames: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    aligned_faces: list[torch.Tensor] = []
    input_mask_list: list[bool] = []

    if sample_frames.numel() == 0:
        emb_dim = int(getattr(embedder, "embedding_size", 512))
        return torch.empty((0, emb_dim), device=device), torch.empty((0,), dtype=torch.bool, device=device), aligned_faces

    for frame in sample_frames:
        detections = detector.detect(frame)
        if detections:
            top = max(detections, key=lambda d: d.score)
            aligned = aligner.align(frame, top).to(device)
            aligned_faces.append(aligned)
            input_mask_list.append(True)
        else:
            aligned_faces.append(torch.empty(0, device=device))
            input_mask_list.append(False)

    input_mask = torch.tensor(input_mask_list, dtype=torch.bool, device=device)
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
    return full_embeds, input_mask, aligned_faces


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


def main() -> None:
    args = parse_args()
    configure_logging()
    device = torch.device(args.device)
    enabled_metrics = _parse_metrics(args.metrics)
    detection_branch_enabled = bool(enabled_metrics & {"detection", "anonymization", "synchronism", "differentiation", "geometric"})
    diversity_enabled = "diversity" in enabled_metrics
    lpips_enabled = "lpips" in enabled_metrics
    ssim_enabled = "ssim" in enabled_metrics
    perceptual_enabled = lpips_enabled or ssim_enabled

    if args.max_identities is not None and args.max_identities <= 0:
        raise ValueError("--max_identities must be > 0 when provided")
    if args.max_videos_per_identity is not None and args.max_videos_per_identity <= 0:
        raise ValueError("--max_videos_per_identity must be > 0 when provided")
    if args.identities_per_batch <= 0:
        raise ValueError("--identities_per_batch must be > 0")
    if args.keys_per_batch <= 0:
        raise ValueError("--keys_per_batch must be > 0")

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

    key_names = [f"key{i}" for i in range(1, num_keys + 1)]
    required_detection_key = f"key{args.detection_key}"
    perceptual_key_name = required_detection_key if args.inferred_nested_keys else "key1"
    if detection_branch_enabled and required_detection_key not in key_names:
        raise ValueError(f"--detection_key must be in [1, {num_keys}]")
    if perceptual_enabled and perceptual_key_name not in key_names:
        raise ValueError(f"Perceptual metric key must be available in [1, {num_keys}]")

    _log_pipe(
        "evaluation_start",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        num_entries=len(entries),
        num_keys=num_keys,
        detection_key=required_detection_key,
        enabled_metrics=",".join(sorted(enabled_metrics)),
        compute_auc_eer=bool(args.compute_auc_eer),
        anonymization_threshold=float(args.ano_threshold),
        synchronism_threshold=float(args.syn_threshold),
        diversity_threshold=float(args.div_threshold),
        differentiation_threshold=float(args.diff_threshold),
        lpips_net=args.lpips_net,
        lpips_cache_dir=str(args.lpips_cache_dir) if args.lpips_cache_dir is not None else None,
        identities_per_batch=int(args.identities_per_batch),
        keys_per_batch=int(args.keys_per_batch),
    )

    detector = MTCNNDetector(
        image_size=256,
        margin=0,
        score_threshold=0.4,
        min_face_size=20,
        max_faces=None,
        keep_all=True,
        post_process=False,
        device=str(device),
    )
    aligner = MTCNNAligner(output_size=256)
    embedder = FacenetEmbedder(pretrained="vggface2", device=str(device))

    metrics = MetricsAccumulator(
        anonymization_threshold=args.ano_threshold,
        synchronism_threshold=args.syn_threshold,
        diversity_threshold=args.div_threshold,
        differentiation_threshold=args.diff_threshold,
        compute_auc_eer=args.compute_auc_eer,
        anonymization_enabled="anonymization" in enabled_metrics,
        diversity_enabled=diversity_enabled,
    )
    geometric_evaluator = GeometricUtilityEvaluator()
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

    identity_to_label: dict[str, int] = {}
    batch_processing_start_time = time.perf_counter()
    total_samples = 0

    diff_embeddings_by_label: dict[int, list[torch.Tensor]] = {}
    diversity_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    try:
        with torch.no_grad():
            total_entries = len(entries)
            progress = tqdm(entries, total=total_entries, desc="Evaluating entries", unit="entry")
            for entry_idx, entry in enumerate(progress, start=1):
                progress.set_postfix_str(f"identity={entry.identity}")
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
                if diversity_enabled and len(available_key_names) < 2:
                    raise ValueError(
                        f"Diversity is enabled but entry {entry.input_video} has fewer than 2 outputs"
                    )

                input_frames = _load_video_tensor(entry.input_video, device)
                if input_frames.shape[0] == 0:
                    continue

                real_embeddings, input_mask, input_faces = _collect_detected_faces(
                    detector,
                    aligner,
                    embedder,
                    input_frames,
                    device,
                )

                key_results: dict[str, tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = {}
                eval_key_names = _sorted_key_names(available_key_names)
                for key_name in eval_key_names:
                    output_path = entry.outputs[key_name]
                    output_frames = _load_video_tensor(output_path, device)
                    if output_frames.shape[0] == 0:
                        virt = torch.zeros_like(real_embeddings)
                        gmask = torch.zeros_like(input_mask)
                        gfaces = [torch.empty(0, device=device) for _ in range(int(input_mask.shape[0]))]
                        key_results[key_name] = (virt, gmask, gfaces)
                        continue

                    min_len = min(int(input_frames.shape[0]), int(output_frames.shape[0]))
                    output_frames = output_frames[:min_len]
                    if real_embeddings.shape[0] != min_len:
                        real_embeddings = real_embeddings[:min_len]
                        input_mask = input_mask[:min_len]
                        input_faces = input_faces[:min_len]

                    if perceptual_enabled and key_name == perceptual_key_name and perceptual_evaluator is not None:
                        aligned_inputs, aligned_outputs = perceptual_evaluator.prepare_video_pair(input_frames, output_frames)
                        pair_count = int(aligned_inputs.shape[0])
                        for frame_idx in range(pair_count):
                            lpips_value, ssim_value = perceptual_evaluator.compute_frame_pair(
                                aligned_inputs[frame_idx],
                                aligned_outputs[frame_idx],
                            )
                            metrics.update_perceptual_utility(lpips_value, ssim_value)

                    virt_embeddings, gen_mask, gen_faces = _collect_detected_faces(
                        detector,
                        aligner,
                        embedder,
                        output_frames,
                        device,
                    )
                    key_results[key_name] = (virt_embeddings, gen_mask, gen_faces)

                det_embeddings = torch.empty((0, 0), device=device)
                det_gen_mask = torch.empty((0,), dtype=torch.bool, device=device)
                valid_mask = torch.empty((0,), dtype=torch.bool, device=device)
                if detection_branch_enabled:
                    det_embeddings, det_gen_mask, _ = key_results[required_detection_key]
                    valid_mask = input_mask & det_gen_mask

                if "detection" in enabled_metrics:
                    metrics.update_detection(det_gen_mask)
                if "anonymization" in enabled_metrics:
                    metrics.update_anonymization(real_embeddings, det_embeddings, valid_mask)

                valid_virtual = det_embeddings[valid_mask] if detection_branch_enabled else torch.empty((0, 0), device=device)
                if "synchronism" in enabled_metrics and valid_virtual.numel() > 0:
                    metrics.add_synchronism_embeddings(label, valid_virtual, source_id=entry.source_id)
                if "differentiation" in enabled_metrics and valid_virtual.numel() > 0:
                    diff_embeddings_by_label.setdefault(label, []).append(valid_virtual.detach().to("cpu"))

                for key_a, key_b in itertools.combinations(eval_key_names, 2):
                    emb_a, mask_a, _ = key_results[key_a]
                    emb_b, mask_b, _ = key_results[key_b]
                    pair_mask = input_mask & mask_a & mask_b
                    if diversity_enabled and pair_mask.any():
                        diversity_pairs.append((emb_a[pair_mask].detach().to("cpu"), emb_b[pair_mask].detach().to("cpu")))

                if "geometric" in enabled_metrics:
                    for key_name in eval_key_names:
                        _, _, gen_faces = key_results[key_name]
                        pair_count = min(len(input_faces), len(gen_faces))
                        for frame_idx in range(pair_count):
                            input_face = input_faces[frame_idx]
                            generated_face = gen_faces[frame_idx]
                            if input_face.numel() == 0 or generated_face.numel() == 0:
                                metrics.update_geometric_utility(None, None)
                            else:
                                errs = geometric_evaluator.compute_pair_errors(
                                    _to_uint8_rgb_image(input_face),
                                    _to_uint8_rgb_image(generated_face),
                                )
                                metrics.update_geometric_utility(errs.head_posture_error, errs.facial_expression_error)

                total_samples += 1

        if "diversity" in enabled_metrics and diversity_pairs:
            _log_pipe(
                "diversity_aggregation_start",
                pair_groups=len(diversity_pairs),
                keys_per_batch=int(args.keys_per_batch),
            )
            diversity_progress = tqdm(diversity_pairs, desc="Aggregating diversity", unit="pair")
            try:
                for emb_a_cpu, emb_b_cpu in diversity_progress:
                    metrics.update_diversity(emb_a_cpu, emb_b_cpu, chunk_size=args.keys_per_batch)
            finally:
                diversity_progress.close()
            _log_pipe(
                "diversity_aggregation_end",
                pair_groups=len(diversity_pairs),
            )

        if "differentiation" in enabled_metrics and diff_embeddings_by_label:
            _log_pipe(
                "differentiation_aggregation_start",
                identities=len(diff_embeddings_by_label),
                identities_per_batch=int(args.identities_per_batch),
                keys_per_batch=int(args.keys_per_batch),
            )
            metrics.update_differentiation_batched(
                diff_embeddings_by_label,
                identities_per_batch=args.identities_per_batch,
                key_chunk_size=args.keys_per_batch,
                show_progress=True,
                progress_desc="Aggregating differentiation",
            )
            _log_pipe(
                "differentiation_aggregation_end",
                identities=len(diff_embeddings_by_label),
            )

    finally:
        geometric_evaluator.close()

    batch_processing_end_time = time.perf_counter()
    batch_processing_seconds = max(0.0, batch_processing_end_time - batch_processing_start_time)

    summary = metrics.finalize()
    _mask_disabled_metrics(summary, enabled_metrics)
    _log_pipe(
        "evaluation_summary",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        total_samples=total_samples,
        batch_processing_seconds=batch_processing_seconds,
        detection_rate=summary["detection_rate"],
    )
    _log_pipe(
        "metric_anonymization",
        success_rate=summary["anonymization"]["success_rate"],
        threshold=summary["anonymization"]["threshold"],
        auc=summary["anonymization"]["auc"],
        eer=summary["anonymization"]["eer"],
        eer_threshold=summary["anonymization"]["eer_threshold"],
        success=summary["anonymization"]["counts"]["success"],
        total=summary["anonymization"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_total",
        success_rate=summary["synchronism_total"]["success_rate"],
        threshold=summary["synchronism_total"]["threshold"],
        auc=summary["synchronism_total"]["auc"],
        eer=summary["synchronism_total"]["eer"],
        eer_threshold=summary["synchronism_total"]["eer_threshold"],
        success=summary["synchronism_total"]["counts"]["success"],
        total=summary["synchronism_total"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_within",
        success_rate=summary["synchronism_within"]["success_rate"],
        threshold=summary["synchronism_within"]["threshold"],
        auc=summary["synchronism_within"]["auc"],
        eer=summary["synchronism_within"]["eer"],
        eer_threshold=summary["synchronism_within"]["eer_threshold"],
        success=summary["synchronism_within"]["counts"]["success"],
        total=summary["synchronism_within"]["counts"]["total"],
    )
    _log_pipe(
        "metric_synchronism_cross",
        success_rate=summary["synchronism_cross"]["success_rate"],
        threshold=summary["synchronism_cross"]["threshold"],
        auc=summary["synchronism_cross"]["auc"],
        eer=summary["synchronism_cross"]["eer"],
        eer_threshold=summary["synchronism_cross"]["eer_threshold"],
        success=summary["synchronism_cross"]["counts"]["success"],
        total=summary["synchronism_cross"]["counts"]["total"],
    )
    _log_pipe(
        "metric_diversity",
        success_rate=summary["diversity"]["success_rate"],
        threshold=summary["diversity"]["threshold"],
        auc=summary["diversity"]["auc"],
        eer=summary["diversity"]["eer"],
        eer_threshold=summary["diversity"]["eer_threshold"],
        success=summary["diversity"]["counts"]["success"],
        total=summary["diversity"]["counts"]["total"],
    )
    _log_pipe(
        "metric_differentiation",
        success_rate=summary["differentiation"]["success_rate"],
        threshold=summary["differentiation"]["threshold"],
        auc=summary["differentiation"]["auc"],
        eer=summary["differentiation"]["eer"],
        eer_threshold=summary["differentiation"]["eer_threshold"],
        success=summary["differentiation"]["counts"]["success"],
        total=summary["differentiation"]["counts"]["total"],
    )
    _log_pipe(
        "metric_geometric_utility",
        head_posture_error=summary["geometric_utility"]["head_posture_error"],
        facial_expression_error=summary["geometric_utility"]["facial_expression_error"],
        valid_pairs=summary["geometric_utility"]["counts"]["valid_pairs"],
        invalid_pairs=summary["geometric_utility"]["counts"]["invalid_pairs"],
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
