#!/usr/bin/env python3
"""
Emotion Recognition Script for RAVDESS Dataset.
"""


import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Initialize PyTorch CUDA BEFORE tensorflow gets imported. If TF claims
# the CUDA context first (during .h5 model load), torch's subsequent
# .to('cuda') in RetinaFace segfaults inside torch._C._cuda_init.
# Forcing a real CUDA op here makes torch own the context cleanly first.
import torch
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device='cuda')
    del _

import faulthandler
faulthandler.enable()

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from ravdess_utils import collect_ravdess_videos, collect_keyed_ravdess_videos, emotion_to_one_hot
    from emotion_metrics import (
        VideoEvaluationResult,
        calculate_absolute_confidence_shift_and_flip_rate,
        calculate_different_key_pairwise_metrics,
        calculate_same_key_better_pairs,
        calculate_same_key_pairwise_metrics,
        generate_metrics_report,
    )
    from emotion_inference import EmotionInferenceEngine
else:
    from .ravdess_utils import collect_ravdess_videos, collect_keyed_ravdess_videos, emotion_to_one_hot
    from emotion_metrics import (
        VideoEvaluationResult,
        calculate_absolute_confidence_shift_and_flip_rate,
        calculate_different_key_pairwise_metrics,
        calculate_same_key_better_pairs,
        calculate_same_key_pairwise_metrics,
        generate_metrics_report,
    )
    from .emotion_inference import EmotionInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_inputs(args) -> bool:
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory is invalid: {input_dir}")
        return False
    if not Path(args.backbone_checkpoint).exists():
        logger.error(f"Backbone checkpoint not found: {args.backbone_checkpoint}")
        return False
    if not Path(args.lstm_checkpoint).exists():
        logger.error(f"LSTM checkpoint not found: {args.lstm_checkpoint}")
        return False
    if args.inferred_keyed_dir:
        if args.num_keys is None or args.num_keys < 1:
            logger.error("--num-keys must be provided and >= 1 when --inferred-keyed-dir is set")
            return False
        for key_index in range(1, args.num_keys + 1):
            key_dir = input_dir / f"key{key_index}"
            if not key_dir.exists() or not key_dir.is_dir():
                logger.error(f"Missing keyed directory: {key_dir}")
                return False
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotion recognition evaluation on RAVDESS dataset")
    parser.add_argument("--input-dir", required=True, help="Directory containing RAVDESS videos")
    parser.add_argument("--backbone-checkpoint", required=True, help="Path to ResNet50 backbone checkpoint")
    parser.add_argument("--lstm-checkpoint", required=True, help="Path to LSTM checkpoint")
    parser.add_argument("--output-json", required=True, help="Path to output JSON report")
    parser.add_argument("--inferred-keyed-dir", action="store_true", help="Treat input-dir as containing key subfolders")
    parser.add_argument("--num-keys", type=int, help="Number of key subfolders (key1..keyN)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Face detection confidence threshold")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    parser.add_argument(
        "--filename-prefix",
        default="video_sample{n}_",
        help="Filename prefix template; use {n} for sample id digits",
    )
    args = parser.parse_args()

    logger.info("Starting emotion recognition evaluation")
    logger.info("Input directory: %s", args.input_dir)
    logger.info("Backbone checkpoint: %s", args.backbone_checkpoint)
    logger.info("LSTM checkpoint: %s", args.lstm_checkpoint)
    logger.info("Output JSON: %s", args.output_json)
    logger.info("Confidence threshold: %.3f", args.confidence_threshold)
    logger.info("Device: %s", args.device)
    logger.info("Filename prefix template: %s", args.filename_prefix)
    logger.info("Inferred keyed dir: %s", args.inferred_keyed_dir)
    if args.inferred_keyed_dir:
        logger.info("Num keys: %s", args.num_keys)

    if not validate_inputs(args):
        return 1

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)

    keyed_metadata = {}
    failed_files = []
    all_video_entries = []
    if args.inferred_keyed_dir:
        keyed_metadata, failed_files = collect_keyed_ravdess_videos(
            input_dir,
            args.num_keys,
            args.filename_prefix,
        )
        for key_label, metadata_dict in keyed_metadata.items():
            for video_path, metadata in metadata_dict.items():
                all_video_entries.append((video_path, metadata, key_label))
    else:
        metadata_dict, failed_files = collect_ravdess_videos(
            input_dir,
            args.filename_prefix,
        )
        keyed_metadata = {"unkeyed": metadata_dict}
        for video_path, metadata in metadata_dict.items():
            all_video_entries.append((video_path, metadata, None))

    logger.info("Collected %d total video entries", len(all_video_entries))
    if failed_files:
        logger.warning("Failed to parse %d files", len(failed_files))

    if not all_video_entries:
        logger.error("No valid RAVDESS videos found")
        return 1

    entries_by_actor = {}
    for video_path, metadata, key_label in all_video_entries:
        entries_by_actor.setdefault(metadata.actor, []).append((video_path, metadata, key_label))

    logger.info("Grouped entries by actor: %d actors", len(entries_by_actor))

    inference_engine = EmotionInferenceEngine(
        backbone_checkpoint=args.backbone_checkpoint,
        lstm_checkpoint=args.lstm_checkpoint,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    logger.info("Inference engine initialized")

    video_results = []
    failed_videos = []
    for actor, actor_entries in tqdm(entries_by_actor.items(), total=len(entries_by_actor)):
        logger.info("Processing actor %s (%d videos)", actor, len(actor_entries))
        for video_path, metadata, key_label in actor_entries:
            result = inference_engine.process_video(video_path)
            if not result["success"]:
                failed_videos.append({
                    "filename": metadata.filename,
                    "key": key_label,
                    "reason": result["error"],
                })
                continue

            avg_probs = result["average_probabilities"]
            pred_emotion = EmotionInferenceEngine.EMOTION_CLASSES[result["predicted_class_idx"]]
            gt_one_hot = emotion_to_one_hot(metadata.emotion_label)
            brier_score = float(np.mean((np.array(avg_probs) - np.array(gt_one_hot)) ** 2))

            video_results.append(
                VideoEvaluationResult(
                    filepath=video_path,
                    metadata=metadata,
                    predicted_emotion=pred_emotion,
                    predicted_probabilities=avg_probs,
                    ground_truth_emotion=metadata.emotion_label,
                    is_correct=pred_emotion == metadata.emotion_label,
                    brier_score=brier_score,
                    key_label=key_label,
                )
            )

    metrics_report_combined = generate_metrics_report(video_results)
    results_by_key = {}
    if args.inferred_keyed_dir:
        for key_label in keyed_metadata.keys():
            results_by_key[key_label] = [r for r in video_results if r.key_label == key_label]
    else:
        results_by_key = {"unkeyed": video_results}

    per_key_uar_brier = {}
    for key_label, key_results in results_by_key.items():
        key_report = generate_metrics_report(key_results)
        per_key_uar_brier[key_label] = {
            "unweighted_average_recall_percent": round(key_report.unweighted_average_recall, 2),
            "brier_score": round(key_report.brier_score, 6),
        }

    key_a = "key1"
    key_b = "key2"
    if results_by_key:
        same_key_agreement, same_key_pairs, same_key_cond, same_key_cond_pairs = (
            calculate_same_key_pairwise_metrics(results_by_key)
        )
    else:
        same_key_agreement = 0.0
        same_key_pairs = 0
        same_key_cond = 0.0
        same_key_cond_pairs = 0

    if args.inferred_keyed_dir and key_a in results_by_key and key_b in results_by_key:
        absolute_confidence_shift, label_flip_rate, matched_pairs, flip_filenames = (
            calculate_absolute_confidence_shift_and_flip_rate(results_by_key, key_a, key_b)
        )
        diff_key_agreement, diff_key_pairs, diff_key_cond, diff_key_cond_pairs = (
            calculate_different_key_pairwise_metrics(video_results)
        )
        same_key_better_agreement, same_key_better_conditional = calculate_same_key_better_pairs(
            results_by_key,
            key_a,
            key_b,
        )
    else:
        absolute_confidence_shift = 0.0
        label_flip_rate = 0.0
        matched_pairs = 0
        flip_filenames = []
        diff_key_agreement = 0.0
        diff_key_pairs = 0
        diff_key_cond = 0.0
        diff_key_cond_pairs = 0
        same_key_better_agreement = []
        same_key_better_conditional = []

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "total_videos_found": len(all_video_entries),
            "videos_processed": len(video_results),
            "videos_failed": len(failed_videos),
            "inferred_keyed_dir": args.inferred_keyed_dir,
            "num_keys": args.num_keys if args.inferred_keyed_dir else None,
            "model_checkpoints": {
                "backbone": str(args.backbone_checkpoint),
                "lstm": str(args.lstm_checkpoint),
            },
            "inference_parameters": {
                "confidence_threshold": args.confidence_threshold,
                "device": args.device,
            },
        },
        "overall_metrics": {
            "uar_brier": {
                "combined": {
                    "unweighted_average_recall_percent": round(
                        metrics_report_combined.unweighted_average_recall,
                        2,
                    ),
                    "brier_score": round(metrics_report_combined.brier_score, 6),
                },
                "per_key": per_key_uar_brier,
            },
            "absolute_confidence_shift": {
                "average": round(absolute_confidence_shift, 6),
                "pair_count": matched_pairs,
                "key_a": key_a if args.inferred_keyed_dir else None,
                "key_b": key_b if args.inferred_keyed_dir else None,
            },
            "label_flip_rate": {
                "rate": round(label_flip_rate, 6),
                "pair_count": matched_pairs,
                "key_a": key_a if args.inferred_keyed_dir else None,
                "key_b": key_b if args.inferred_keyed_dir else None,
            },
            "pairwise_agreement_rate": {
                "same_key": round(same_key_agreement, 6),
                "same_key_pairs": same_key_pairs,
                "different_key": round(diff_key_agreement, 6),
                "different_key_pairs": diff_key_pairs,
            },
            "conditional_accuracy": {
                "same_key": round(same_key_cond, 6),
                "same_key_pairs": same_key_cond_pairs,
                "different_key": round(diff_key_cond, 6),
                "different_key_pairs": diff_key_cond_pairs,
            },
            "label_flip_filenames": flip_filenames,
            "same_key_better_agreement_pairs": same_key_better_agreement,
            "same_key_better_conditional_pairs": same_key_better_conditional,
        },
        "video_results": [],
        "failed_videos": failed_videos,
    }

    logger.info(
        "Metrics computed: brier=%.6f, uar=%.2f",
        metrics_report_combined.brier_score,
        metrics_report_combined.unweighted_average_recall,
    )

    for result in video_results:
        report["video_results"].append(
            {
                "filename": result.metadata.filename,
                "key": result.key_label,
                "actor": result.metadata.actor,
                "emotion_ground_truth": result.ground_truth_emotion,
                "emotion_predicted": result.predicted_emotion,
                "predicted_probabilities": {
                    emotion: round(prob, 4)
                    for emotion, prob in zip(
                        ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"],
                        result.predicted_probabilities,
                    )
                },
                "is_correct": result.is_correct,
                "brier_score": round(result.brier_score, 6),
            }
        )

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info("Report saved to %s", output_json)
    logger.info("Combined Brier Score: %.6f", metrics_report_combined.brier_score)
    logger.info("Combined UAR: %.2f%%", metrics_report_combined.unweighted_average_recall)
    if args.inferred_keyed_dir and key_a in results_by_key and key_b in results_by_key:
        logger.info("Absolute confidence shift (%s vs %s): %.6f", key_a, key_b, absolute_confidence_shift)
        logger.info("Label flip rate (%s vs %s): %.6f", key_a, key_b, label_flip_rate)
        logger.info("Pairwise agreement (same-key): %.6f", same_key_agreement)
        logger.info("Pairwise agreement (different-key): %.6f", diff_key_agreement)
        logger.info("Conditional accuracy (same-key): %.6f", same_key_cond)
        logger.info("Conditional accuracy (different-key): %.6f", diff_key_cond)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
