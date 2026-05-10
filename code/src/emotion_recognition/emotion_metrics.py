"""
Emotion recognition metrics calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from .ravdess_utils import RAVDESSMetadata, emotion_to_one_hot, EMOTION_TO_IDX
except ImportError:  # Allow direct script execution without package context.
    from ravdess_utils import RAVDESSMetadata, emotion_to_one_hot, EMOTION_TO_IDX


@dataclass
class VideoEvaluationResult:
    filepath: str
    metadata: RAVDESSMetadata
    predicted_emotion: str
    predicted_probabilities: List[float]
    ground_truth_emotion: str
    is_correct: bool
    brier_score: float
    key_label: Optional[str] = None


@dataclass
class MetricsReport:
    classification_accuracy: float
    brier_score: float
    unweighted_average_recall: float
    pair_consistency: float
    pair_consistency_pair_count: int
    per_actor_accuracy: Dict[str, float]
    per_emotion_accuracy: Dict[str, float]
    per_intensity_accuracy: Dict[str, float]
    video_results: List[VideoEvaluationResult]


def calculate_classification_accuracy(
    video_results: List[VideoEvaluationResult],
    by_group: Optional[List[str]] = None,
) -> Dict[str, float]:
    results = [r for r in video_results if by_group is None or r.filepath in by_group]
    if not results:
        return {"overall": 0.0}
    correct = sum(1 for r in results if r.is_correct)
    return {"overall": (correct / len(results)) * 100}


def calculate_brier_score(
    video_results: List[VideoEvaluationResult],
    by_group: Optional[List[str]] = None,
) -> float:
    results = [r for r in video_results if by_group is None or r.filepath in by_group]
    if not results:
        return 0.0
    scores = []
    for result in results:
        gt_one_hot = np.array(emotion_to_one_hot(result.ground_truth_emotion), dtype=np.float32)
        pred_probs = np.array(result.predicted_probabilities, dtype=np.float32)
        scores.append(float(np.mean((pred_probs - gt_one_hot) ** 2)))
    return float(np.mean(scores))


def calculate_unweighted_average_recall(
    video_results: List[VideoEvaluationResult],
    by_group: Optional[List[str]] = None,
) -> float:
    results = [r for r in video_results if by_group is None or r.filepath in by_group]
    if not results:
        return 0.0

    per_class_recalls = []
    class_labels = sorted({r.ground_truth_emotion for r in results})
    for label in class_labels:
        label_results = [r for r in results if r.ground_truth_emotion == label]
        if not label_results:
            continue
        true_positives = sum(1 for r in label_results if r.predicted_emotion == label)
        per_class_recalls.append(true_positives / len(label_results))

    return float(np.mean(per_class_recalls) * 100) if per_class_recalls else 0.0


def calculate_pair_consistency(
    video_results: List[VideoEvaluationResult],
    by_group: Optional[List[str]] = None,
) -> Tuple[float, int]:
    results = [r for r in video_results if by_group is None or r.filepath in by_group]
    groups: Dict[Tuple[str, str, str], List[VideoEvaluationResult]] = {}
    for result in results:
        group_key = (result.metadata.actor, result.ground_truth_emotion, result.metadata.intensity)
        groups.setdefault(group_key, []).append(result)

    total_diff = 0.0
    pair_count = 0
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            prob_i = group_results[i].predicted_probabilities[EMOTION_TO_IDX[group_results[i].ground_truth_emotion]]
            for j in range(i + 1, len(group_results)):
                prob_j = group_results[j].predicted_probabilities[EMOTION_TO_IDX[group_results[j].ground_truth_emotion]]
                total_diff += abs(prob_i - prob_j)
                pair_count += 1

    return (total_diff / pair_count if pair_count else 0.0), pair_count


def calculate_same_key_pair_consistency(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
) -> Tuple[float, int]:
    total_diff = 0.0
    total_pairs = 0
    for key_results in results_by_key.values():
        avg_diff, pair_count = calculate_pair_consistency(key_results)
        total_diff += avg_diff * pair_count
        total_pairs += pair_count
    return (total_diff / total_pairs if total_pairs else 0.0), total_pairs


def calculate_different_key_pair_consistency(
    video_results: List[VideoEvaluationResult],
) -> Tuple[float, int]:
    groups: Dict[Tuple[str, str, str], List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = (result.metadata.actor, result.ground_truth_emotion, result.metadata.intensity)
        groups.setdefault(group_key, []).append(result)

    total_diff = 0.0
    pair_count = 0
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            key_i = group_results[i].key_label
            if key_i is None:
                continue
            prob_i = group_results[i].predicted_probabilities[EMOTION_TO_IDX[group_results[i].ground_truth_emotion]]
            for j in range(i + 1, len(group_results)):
                key_j = group_results[j].key_label
                if key_j is None or key_i == key_j:
                    continue
                prob_j = group_results[j].predicted_probabilities[EMOTION_TO_IDX[group_results[j].ground_truth_emotion]]
                total_diff += abs(prob_i - prob_j)
                pair_count += 1

    return (total_diff / pair_count if pair_count else 0.0), pair_count


def aggregate_by_actor(video_results: List[VideoEvaluationResult]) -> Dict[str, Dict]:
    by_actor = {}
    for result in video_results:
        by_actor.setdefault(result.metadata.actor, []).append(result)

    result_dict = {}
    for actor, actor_results in by_actor.items():
        filepaths = [r.filepath for r in actor_results]
        result_dict[actor] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "video_count": len(filepaths),
            "gender": actor_results[0].metadata.actor_gender,
        }
    return result_dict


def aggregate_by_emotion(video_results: List[VideoEvaluationResult]) -> Dict[str, Dict]:
    by_emotion = {}
    for result in video_results:
        by_emotion.setdefault(result.metadata.emotion_label, []).append(result)

    result_dict = {}
    for emotion, emotion_results in by_emotion.items():
        filepaths = [r.filepath for r in emotion_results]
        result_dict[emotion] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "video_count": len(filepaths),
        }
    return result_dict


def aggregate_by_intensity(video_results: List[VideoEvaluationResult]) -> Dict[str, Dict]:
    by_intensity = {}
    for result in video_results:
        by_intensity.setdefault(result.metadata.intensity_label, []).append(result)

    result_dict = {}
    for intensity, intensity_results in by_intensity.items():
        filepaths = [r.filepath for r in intensity_results]
        result_dict[intensity] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "video_count": len(filepaths),
        }
    return result_dict


def generate_metrics_report(
    video_results: List[VideoEvaluationResult],
) -> MetricsReport:
    overall_accuracy = calculate_classification_accuracy(video_results)["overall"]
    overall_brier_score = calculate_brier_score(video_results)
    overall_uar = calculate_unweighted_average_recall(video_results)
    overall_pair_consistency, pair_count = calculate_pair_consistency(video_results)

    by_actor = aggregate_by_actor(video_results)
    by_emotion = aggregate_by_emotion(video_results)
    by_intensity = aggregate_by_intensity(video_results)

    return MetricsReport(
        classification_accuracy=overall_accuracy,
        brier_score=overall_brier_score,
        unweighted_average_recall=overall_uar,
        pair_consistency=overall_pair_consistency,
        pair_consistency_pair_count=pair_count,
        per_actor_accuracy={actor: data["accuracy"] for actor, data in by_actor.items()},
        per_emotion_accuracy={emotion: data["accuracy"] for emotion, data in by_emotion.items()},
        per_intensity_accuracy={intensity: data["accuracy"] for intensity, data in by_intensity.items()},
        video_results=video_results,
    )
