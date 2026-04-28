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
    from ravdess_utils import collect_ravdess_videos, validate_repetition_pairs, emotion_to_one_hot
    from emotion_metrics import VideoEvaluationResult, generate_metrics_report
    from emotion_inference import EmotionInferenceEngine
else:
    from .ravdess_utils import collect_ravdess_videos, validate_repetition_pairs, emotion_to_one_hot
    from .emotion_metrics import VideoEvaluationResult, generate_metrics_report
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
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotion recognition evaluation on RAVDESS dataset")
    parser.add_argument("--input-dir", required=True, help="Directory containing RAVDESS videos")
    parser.add_argument("--backbone-checkpoint", required=True, help="Path to ResNet50 backbone checkpoint")
    parser.add_argument("--lstm-checkpoint", required=True, help="Path to LSTM checkpoint")
    parser.add_argument("--output-json", required=True, help="Path to output JSON report")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Face detection confidence threshold")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    args = parser.parse_args()

    if not validate_inputs(args):
        return 1

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)

    metadata_dict, failed_files = collect_ravdess_videos(input_dir)
    if not metadata_dict:
        logger.error("No valid RAVDESS videos found")
        return 1

    repetition_pairs = validate_repetition_pairs(metadata_dict)
    complete_pairs = sum(1 for rep01, rep02 in repetition_pairs.values() if rep01 is not None and rep02 is not None)

    inference_engine = EmotionInferenceEngine(
        backbone_checkpoint=args.backbone_checkpoint,
        lstm_checkpoint=args.lstm_checkpoint,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    video_results = []
    failed_videos = []
    for video_path, metadata in tqdm(metadata_dict.items(), total=len(metadata_dict)):
        result = inference_engine.process_video(video_path)
        if not result["success"]:
            failed_videos.append({"filename": metadata.filename, "reason": result["error"]})
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
            )
        )

    metrics_report = generate_metrics_report(video_results, repetition_pairs)
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "total_videos_found": len(metadata_dict),
            "videos_processed": len(video_results),
            "videos_failed": len(failed_videos),
            "complete_repetition_pairs": complete_pairs,
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
            "repetition_consistency_percent": round(metrics_report.repetition_consistency, 2),
        },
        "per_actor_metrics": {},
        "per_emotion_metrics": {},
        "per_intensity_metrics": {},
        "video_results": [],
        "failed_videos": failed_videos,
    }

    for actor, acc in metrics_report.per_actor_accuracy.items():
        report["per_actor_metrics"][actor] = {
            "accuracy_percent": round(acc, 2),
            "consistency_percent": round(metrics_report.per_actor_consistency[actor], 2),
        }
    for emotion, acc in metrics_report.per_emotion_accuracy.items():
        report["per_emotion_metrics"][emotion] = {
            "accuracy_percent": round(acc, 2),
            "consistency_percent": round(metrics_report.per_emotion_consistency[emotion], 2),
        }
    for intensity, acc in metrics_report.per_intensity_accuracy.items():
        report["per_intensity_metrics"][intensity] = {
            "accuracy_percent": round(acc, 2),
            "consistency_percent": round(metrics_report.per_intensity_consistency[intensity], 2),
        }

    for result in video_results:
        report["video_results"].append(
            {
                "filename": result.metadata.filename,
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
    logger.info("Repetition Consistency: %.2f%%", metrics_report.repetition_consistency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
