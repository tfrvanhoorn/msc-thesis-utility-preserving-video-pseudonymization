"""
Emotion recognition metrics calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .ravdess_utils import RAVDESSMetadata, emotion_to_one_hot


@dataclass
class VideoEvaluationResult:
    filepath: str
    metadata: RAVDESSMetadata
    predicted_emotion: str
    predicted_probabilities: List[float]
    ground_truth_emotion: str
    is_correct: bool
    brier_score: float


@dataclass
class MetricsReport:
    classification_accuracy: float
    brier_score: float
    repetition_consistency: float
    per_actor_accuracy: Dict[str, float]
    per_actor_consistency: Dict[str, float]
    per_emotion_accuracy: Dict[str, float]
    per_emotion_consistency: Dict[str, float]
    per_intensity_accuracy: Dict[str, float]
    per_intensity_consistency: Dict[str, float]
    repetition_pairs_results: Dict[Tuple[str, str, str], Dict]
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


def calculate_repetition_consistency(
    video_results: List[VideoEvaluationResult],
    repetition_pairs: Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]],
    by_group: Optional[List[str]] = None,
) -> Tuple[float, Dict]:
    result_lookup = {r.filepath: r for r in video_results}
    if by_group is not None:
        allowed = set(by_group)
        result_lookup = {k: v for k, v in result_lookup.items() if k in allowed}

    total_pairs = 0
    matching_pairs = 0
    detailed_results = {}

    for (actor, emotion_code, intensity), (rep01_path, rep02_path) in repetition_pairs.items():
        has_rep01 = rep01_path is not None and rep01_path in result_lookup
        has_rep02 = rep02_path is not None and rep02_path in result_lookup
        if not (has_rep01 and has_rep02):
            detailed_results[(actor, emotion_code, intensity)] = {
                "status": "incomplete",
                "rep01_available": has_rep01,
                "rep02_available": has_rep02,
                "match": None,
            }
            continue

        total_pairs += 1
        result_rep01 = result_lookup[rep01_path]
        result_rep02 = result_lookup[rep02_path]
        match = result_rep01.predicted_emotion == result_rep02.predicted_emotion
        if match:
            matching_pairs += 1
        detailed_results[(actor, emotion_code, intensity)] = {
            "status": "complete",
            "rep01_prediction": result_rep01.predicted_emotion,
            "rep02_prediction": result_rep02.predicted_emotion,
            "match": match,
            "rep01_probabilities": result_rep01.predicted_probabilities,
            "rep02_probabilities": result_rep02.predicted_probabilities,
        }

    return ((matching_pairs / total_pairs) * 100 if total_pairs else 0.0), detailed_results


def aggregate_by_actor(
    video_results: List[VideoEvaluationResult],
    repetition_pairs: Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]],
) -> Dict[str, Dict]:
    by_actor = {}
    for result in video_results:
        by_actor.setdefault(result.metadata.actor, []).append(result)

    result_dict = {}
    for actor, actor_results in by_actor.items():
        filepaths = [r.filepath for r in actor_results]
        actor_pairs = {k: v for k, v in repetition_pairs.items() if k[0] == actor}
        cons, _ = calculate_repetition_consistency(video_results, actor_pairs, filepaths)
        result_dict[actor] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "consistency": cons,
            "video_count": len(filepaths),
            "gender": actor_results[0].metadata.actor_gender,
        }
    return result_dict


def aggregate_by_emotion(
    video_results: List[VideoEvaluationResult],
    repetition_pairs: Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]],
) -> Dict[str, Dict]:
    by_emotion = {}
    for result in video_results:
        by_emotion.setdefault(result.metadata.emotion_label, []).append(result)

    result_dict = {}
    for emotion, emotion_results in by_emotion.items():
        filepaths = [r.filepath for r in emotion_results]
        emotion_pairs = {}
        for key, pair in repetition_pairs.items():
            pair_results = [r for r in video_results if r.filepath in pair]
            if pair_results and pair_results[0].metadata.emotion_label == emotion:
                emotion_pairs[key] = pair
        cons, _ = calculate_repetition_consistency(video_results, emotion_pairs, filepaths)
        result_dict[emotion] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "consistency": cons,
            "video_count": len(filepaths),
        }
    return result_dict


def aggregate_by_intensity(
    video_results: List[VideoEvaluationResult],
    repetition_pairs: Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]],
) -> Dict[str, Dict]:
    by_intensity = {}
    for result in video_results:
        by_intensity.setdefault(result.metadata.intensity_label, []).append(result)

    result_dict = {}
    for intensity, intensity_results in by_intensity.items():
        filepaths = [r.filepath for r in intensity_results]
        intensity_pairs = {}
        for key, pair in repetition_pairs.items():
            pair_results = [r for r in video_results if r.filepath in pair]
            if pair_results and pair_results[0].metadata.intensity_label == intensity:
                intensity_pairs[key] = pair
        cons, _ = calculate_repetition_consistency(video_results, intensity_pairs, filepaths)
        result_dict[intensity] = {
            "accuracy": calculate_classification_accuracy(video_results, filepaths)["overall"],
            "consistency": cons,
            "video_count": len(filepaths),
        }
    return result_dict


def generate_metrics_report(
    video_results: List[VideoEvaluationResult],
    repetition_pairs: Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]],
) -> MetricsReport:
    overall_accuracy = calculate_classification_accuracy(video_results)["overall"]
    overall_brier_score = calculate_brier_score(video_results)
    overall_consistency, pair_details = calculate_repetition_consistency(video_results, repetition_pairs)

    by_actor = aggregate_by_actor(video_results, repetition_pairs)
    by_emotion = aggregate_by_emotion(video_results, repetition_pairs)
    by_intensity = aggregate_by_intensity(video_results, repetition_pairs)

    return MetricsReport(
        classification_accuracy=overall_accuracy,
        brier_score=overall_brier_score,
        repetition_consistency=overall_consistency,
        per_actor_accuracy={actor: data["accuracy"] for actor, data in by_actor.items()},
        per_actor_consistency={actor: data["consistency"] for actor, data in by_actor.items()},
        per_emotion_accuracy={emotion: data["accuracy"] for emotion, data in by_emotion.items()},
        per_emotion_consistency={emotion: data["consistency"] for emotion, data in by_emotion.items()},
        per_intensity_accuracy={intensity: data["accuracy"] for intensity, data in by_intensity.items()},
        per_intensity_consistency={intensity: data["consistency"] for intensity, data in by_intensity.items()},
        repetition_pairs_results=pair_details,
        video_results=video_results,
    )
