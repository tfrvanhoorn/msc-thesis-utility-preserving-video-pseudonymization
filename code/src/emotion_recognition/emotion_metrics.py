"""
Emotion recognition metrics calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from .ravdess_utils import (
        RAVDESSMetadata,
        emotion_to_one_hot,
        EMOTION_TO_IDX,
    )
except ImportError:
    from ravdess_utils import (
        RAVDESSMetadata,
        emotion_to_one_hot,
        EMOTION_TO_IDX,
    )


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

def _group_by_actor_emotion(
    video_results: List[VideoEvaluationResult],
) -> Dict[Tuple[str, str], List[VideoEvaluationResult]]:
    groups: Dict[Tuple[str, str], List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = (result.metadata.actor, result.ground_truth_emotion)
        groups.setdefault(group_key, []).append(result)
    return groups


def _build_filename_lookup(
    video_results: List[VideoEvaluationResult],
) -> Dict[str, VideoEvaluationResult]:
    return {result.metadata.filename: result for result in video_results}


def calculate_absolute_confidence_shift_and_flip_rate(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
    key_a: str,
    key_b: str,
) -> Tuple[float, float, int, List[str]]:
    if key_a not in results_by_key or key_b not in results_by_key:
        return 0.0, 0.0, 0, []

    lookup_a = _build_filename_lookup(results_by_key[key_a])
    lookup_b = _build_filename_lookup(results_by_key[key_b])
    matched_filenames = sorted(set(lookup_a.keys()) & set(lookup_b.keys()))

    total_shift = 0.0
    flip_filenames: List[str] = []
    for filename in matched_filenames:
        result_a = lookup_a[filename]
        result_b = lookup_b[filename]
        emotion_idx = EMOTION_TO_IDX[result_a.ground_truth_emotion]
        prob_a = float(result_a.predicted_probabilities[emotion_idx])
        prob_b = float(result_b.predicted_probabilities[emotion_idx])
        total_shift += abs(prob_a - prob_b)
        if result_a.predicted_emotion != result_b.predicted_emotion:
            flip_filenames.append(filename)

    pair_count = len(matched_filenames)
    avg_shift = total_shift / pair_count if pair_count else 0.0
    flip_rate = len(flip_filenames) / pair_count if pair_count else 0.0
    return avg_shift, flip_rate, pair_count, flip_filenames


def calculate_same_key_pairwise_metrics(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
) -> Tuple[float, int, float, int]:
    total_agreement = 0
    total_pairs = 0
    conditional_numerator = 0
    conditional_denominator = 0

    for key_results in results_by_key.values():
        groups = _group_by_actor_emotion(key_results)
        for group_results in groups.values():
            if len(group_results) < 2:
                continue
            for i in range(len(group_results)):
                for j in range(i + 1, len(group_results)):
                    total_pairs += 1
                    if group_results[i].predicted_emotion == group_results[j].predicted_emotion:
                        total_agreement += 1
                    if group_results[i].is_correct:
                        conditional_denominator += 1
                        if group_results[j].is_correct:
                            conditional_numerator += 1

    agreement_rate = total_agreement / total_pairs if total_pairs else 0.0
    conditional_rate = (
        conditional_numerator / conditional_denominator if conditional_denominator else 0.0
    )
    return agreement_rate, total_pairs, conditional_rate, conditional_denominator


def calculate_different_key_pairwise_metrics(
    video_results: List[VideoEvaluationResult],
) -> Tuple[float, int, float, int]:
    total_agreement = 0
    total_pairs = 0
    conditional_numerator = 0
    conditional_denominator = 0

    groups = _group_by_actor_emotion(video_results)
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            key_i = group_results[i].key_label
            if key_i is None:
                continue
            for j in range(len(group_results)):
                if i == j:
                    continue
                key_j = group_results[j].key_label
                if key_j is None or key_i == key_j:
                    continue
                if group_results[i].metadata.filename == group_results[j].metadata.filename:
                    continue
                total_pairs += 1
                if group_results[i].predicted_emotion == group_results[j].predicted_emotion:
                    total_agreement += 1
                if group_results[i].is_correct:
                    conditional_denominator += 1
                    if group_results[j].is_correct:
                        conditional_numerator += 1

    agreement_rate = total_agreement / total_pairs if total_pairs else 0.0
    conditional_rate = (
        conditional_numerator / conditional_denominator if conditional_denominator else 0.0
    )
    return agreement_rate, total_pairs, conditional_rate, conditional_denominator


def calculate_same_key_better_pairs(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
    key_a: str,
    key_b: str,
) -> Tuple[List[str], List[str]]:
    if key_a not in results_by_key or key_b not in results_by_key:
        return [], []

    lookup_a = _build_filename_lookup(results_by_key[key_a])
    lookup_b = _build_filename_lookup(results_by_key[key_b])
    common_filenames = sorted(set(lookup_a.keys()) & set(lookup_b.keys()))

    filenames_by_group: Dict[Tuple[str, str], List[str]] = {}
    for filename in common_filenames:
        meta = lookup_a[filename].metadata
        group_key = (meta.actor, meta.emotion_label)
        filenames_by_group.setdefault(group_key, []).append(filename)

    agreement_better: List[str] = []
    conditional_better: List[str] = []

    for filenames in filenames_by_group.values():
        if len(filenames) < 2:
            continue
        filenames = sorted(filenames)
        for i in range(len(filenames)):
            for j in range(len(filenames)):
                if i == j:
                    continue
                fname_i = filenames[i]
                fname_j = filenames[j]
                res_a_i = lookup_a[fname_i]
                res_a_j = lookup_a[fname_j]
                res_b_i = lookup_b[fname_i]
                res_b_j = lookup_b[fname_j]

                same_agreement_vals = [
                    1 if res_a_i.predicted_emotion == res_a_j.predicted_emotion else 0,
                    1 if res_b_i.predicted_emotion == res_b_j.predicted_emotion else 0,
                ]
                same_agreement = float(np.mean(same_agreement_vals))

                diff_agreement_ab = 1 if res_a_i.predicted_emotion == res_b_j.predicted_emotion else 0
                diff_agreement_ba = 1 if res_b_i.predicted_emotion == res_a_j.predicted_emotion else 0

                if same_agreement > diff_agreement_ab:
                    agreement_better.append(
                        f"{fname_i}|{fname_j} (key_order={key_a}->{key_b})"
                    )
                if same_agreement > diff_agreement_ba:
                    agreement_better.append(
                        f"{fname_i}|{fname_j} (key_order={key_b}->{key_a})"
                    )

                same_cond_vals: List[int] = []
                if res_a_i.is_correct:
                    same_cond_vals.append(1 if res_a_j.is_correct else 0)
                if res_b_i.is_correct:
                    same_cond_vals.append(1 if res_b_j.is_correct else 0)

                diff_cond_ab_vals: List[int] = []
                if res_a_i.is_correct:
                    diff_cond_ab_vals.append(1 if res_b_j.is_correct else 0)

                diff_cond_ba_vals: List[int] = []
                if res_b_i.is_correct:
                    diff_cond_ba_vals.append(1 if res_a_j.is_correct else 0)

                if same_cond_vals:
                    same_cond = float(np.mean(same_cond_vals))
                    if diff_cond_ab_vals:
                        diff_cond_ab = float(np.mean(diff_cond_ab_vals))
                        if same_cond > diff_cond_ab:
                            conditional_better.append(
                                f"{fname_i}|{fname_j} (key_order={key_a}->{key_b})"
                            )
                    if diff_cond_ba_vals:
                        diff_cond_ba = float(np.mean(diff_cond_ba_vals))
                        if same_cond > diff_cond_ba:
                            conditional_better.append(
                                f"{fname_i}|{fname_j} (key_order={key_b}->{key_a})"
                            )

    return agreement_better, conditional_better

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

    by_actor = aggregate_by_actor(video_results)
    by_emotion = aggregate_by_emotion(video_results)
    by_intensity = aggregate_by_intensity(video_results)

    return MetricsReport(
        classification_accuracy=overall_accuracy,
        brier_score=overall_brier_score,
        unweighted_average_recall=overall_uar,
        per_actor_accuracy={actor: data["accuracy"] for actor, data in by_actor.items()},
        per_emotion_accuracy={emotion: data["accuracy"] for emotion, data in by_emotion.items()},
        per_intensity_accuracy={intensity: data["accuracy"] for intensity, data in by_intensity.items()},
        video_results=video_results,
    )
