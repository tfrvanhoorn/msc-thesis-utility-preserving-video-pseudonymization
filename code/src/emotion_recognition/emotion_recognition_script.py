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
        calculate_different_key_pair_consistency,
        calculate_pair_consistency,
        calculate_same_key_pair_consistency,
        generate_metrics_report,
    )
    from emotion_inference import EmotionInferenceEngine
else:
    from .ravdess_utils import collect_ravdess_videos, collect_keyed_ravdess_videos, emotion_to_one_hot
    from .emotion_metrics import (
        VideoEvaluationResult,
        calculate_different_key_pair_consistency,
        calculate_pair_consistency,
        calculate_same_key_pair_consistency,
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
        if args.detection_key is None:
            logger.error("--detection-key must be provided when --inferred-keyed-dir is set")
            return False
        if args.detection_key < 1 or args.detection_key > args.num_keys:
            logger.error("--detection-key must be between 1 and --num-keys")
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
    parser.add_argument("--detection-key", type=int, help="Key index (1..N) for per-clip metrics")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Face detection confidence threshold")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    args = parser.parse_args()

    if not validate_inputs(args):
        return 1

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)

    keyed_metadata = {}
    failed_files = []
    all_video_entries = []
    detection_key_label = None

    if args.inferred_keyed_dir:
        keyed_metadata, failed_files = collect_keyed_ravdess_videos(input_dir, args.num_keys)
        for key_label, metadata_dict in keyed_metadata.items():
            for video_path, metadata in metadata_dict.items():
                all_video_entries.append((video_path, metadata, key_label))
        detection_key_label = f"key{args.detection_key}"
    else:
        metadata_dict, failed_files = collect_ravdess_videos(input_dir)
        keyed_metadata = {"unkeyed": metadata_dict}
        for video_path, metadata in metadata_dict.items():
            all_video_entries.append((video_path, metadata, None))

    if not all_video_entries:
        logger.error("No valid RAVDESS videos found")
        return 1

    entries_by_actor = {}
    for video_path, metadata, key_label in all_video_entries:
        entries_by_actor.setdefault(metadata.actor, []).append((video_path, metadata, key_label))

    inference_engine = EmotionInferenceEngine(
        backbone_checkpoint=args.backbone_checkpoint,
        lstm_checkpoint=args.lstm_checkpoint,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

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
            pred_emotion = inference_engine.get_predicted_emotion(avg_probs)
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

    if args.inferred_keyed_dir:
        detection_video_results = [
            result for result in video_results if result.key_label == detection_key_label
        ]
    else:
        detection_video_results = video_results

    metrics_report = generate_metrics_report(detection_video_results)
    if args.inferred_keyed_dir:
        results_by_key = {}
        for key_label in keyed_metadata.keys():
            results_by_key[key_label] = [r for r in video_results if r.key_label == key_label]
        same_key_consistency, same_key_pairs = calculate_same_key_pair_consistency(results_by_key)
        different_key_consistency, different_key_pairs = calculate_different_key_pair_consistency(video_results)
    else:
        overall_pair_consistency, overall_pair_pairs = calculate_pair_consistency(video_results)
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "total_videos_found": len(all_video_entries),
            "videos_processed": len(video_results),
            "videos_processed_detection_key": len(detection_video_results),
            "videos_failed": len(failed_videos),
            "inferred_keyed_dir": args.inferred_keyed_dir,
            "num_keys": args.num_keys if args.inferred_keyed_dir else None,
            "detection_key": detection_key_label,
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
            "classification_accuracy_percent": round(metrics_report.classification_accuracy, 2),
            "brier_score": round(metrics_report.brier_score, 6),
            "pair_consistency": {},
        },
        "per_actor_metrics": {},
        "per_emotion_metrics": {},
        "per_intensity_metrics": {},
        "video_results": [],
        "failed_videos": failed_videos,
    }

    if args.inferred_keyed_dir:
        report["overall_metrics"]["pair_consistency"] = {
            "same_key_average": round(same_key_consistency, 6),
            "same_key_pairs": same_key_pairs,
            "different_key_average": round(different_key_consistency, 6),
            "different_key_pairs": different_key_pairs,
        }
    else:
        report["overall_metrics"]["pair_consistency"] = {
            "overall_average": round(overall_pair_consistency, 6),
            "overall_pairs": overall_pair_pairs,
        }

    for actor, acc in metrics_report.per_actor_accuracy.items():
        report["per_actor_metrics"][actor] = {
            "accuracy_percent": round(acc, 2),
        }
    for emotion, acc in metrics_report.per_emotion_accuracy.items():
        report["per_emotion_metrics"][emotion] = {
            "accuracy_percent": round(acc, 2),
        }
    for intensity, acc in metrics_report.per_intensity_accuracy.items():
        report["per_intensity_metrics"][intensity] = {
            "accuracy_percent": round(acc, 2),
        }

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
    logger.info("Classification Accuracy: %.2f%%", metrics_report.classification_accuracy)
    logger.info("Brier Score: %.6f", metrics_report.brier_score)
    if args.inferred_keyed_dir:
        logger.info("Pair Consistency (same-key): %.6f", same_key_consistency)
        logger.info("Pair Consistency (different-key): %.6f", different_key_consistency)
    else:
        logger.info("Pair Consistency: %.6f", overall_pair_consistency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
