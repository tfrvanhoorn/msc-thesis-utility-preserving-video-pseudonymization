"""
Emotion recognition metrics calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

import json

try:
    from .ravdess_utils import (
        RAVDESSMetadata,
        emotion_to_one_hot,
        EMOTION_TO_IDX,
        EMOTION_CLASSES,
    )
except ImportError:
    from ravdess_utils import (
        RAVDESSMetadata,
        emotion_to_one_hot,
        EMOTION_TO_IDX,
        EMOTION_CLASSES,
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
    pair_consistency: float
    pair_consistency_pair_count: int
    per_actor_accuracy: Dict[str, float]
    per_emotion_accuracy: Dict[str, float]
    per_intensity_accuracy: Dict[str, float]
    video_results: List[VideoEvaluationResult]

def total_variation_distance(probs_a: List[float], probs_b: List[float]) -> float:
    """TV(P, Q) = 0.5 * sum_c |P(c) - Q(c)|. Bounded in [0, 1]."""
    a = np.asarray(probs_a, dtype=np.float64)
    b = np.asarray(probs_b, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(a - b)))

# ----------------------------------------------------------------------
# Baseline-relative TV deviation (difference-in-differences over pairs).
#
# For each pair of clips (i, j) belonging to the same (actor, emotion,
# intensity) group, the shift vector under condition X is
#     delta_X = p_X(j) - p_X(i)      in R^7, summing to zero.
# The baseline-relative TV deviation is the TV-style L1 distance between
# the pseudonymized shift and the baseline shift,
#     dev(i, j) = 0.5 * sum_c |delta_pseudo,c - delta_baseline,c|.
# A value of 0 means pseudonymization preserves the natural per-pair
# shift exactly; larger values mean pseudonymization adds its own
# per-pair shift on top of the baseline. Bounded in [0, 2] (rather than
# [0, 1] like standard TVD), because delta vectors range over [-1, 1]^7
# instead of the probability simplex.
# ----------------------------------------------------------------------

ClipIdentity = Tuple[str, str, str, str, str]


def _clip_identity(result: VideoEvaluationResult) -> ClipIdentity:
    """(actor, emotion_code, intensity, statement, repetition) — uniquely
    identifies a RAVDESS clip across baseline and pseudonymized runs."""
    meta = result.metadata
    return (meta.actor, meta.emotion_code, meta.intensity, meta.statement, meta.repetition)


def load_baseline_lookup_from_json(
    json_path,
    prefix_template: str = "video_sample{n}_",
) -> Dict[ClipIdentity, List[float]]:
    """Load a previously generated baseline report and return a mapping
    from clip identity to the baseline's predicted probability vector
    (ordered to match EMOTION_CLASSES).

    Assumes the baseline report was produced by this same script with the
    same filename prefix template. Entries that fail to parse are skipped.
    """
    try:
        from .ravdess_utils import parse_ravdess_filename
    except ImportError:
        from ravdess_utils import parse_ravdess_filename

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    lookup: Dict[ClipIdentity, List[float]] = {}
    for entry in data.get("video_results", []):
        filename = entry.get("filename")
        if not filename:
            continue
        parsed = parse_ravdess_filename(filename, prefix_template)
        if parsed is None:
            continue
        identity = (parsed.actor, parsed.emotion_code, parsed.intensity,
                    parsed.statement, parsed.repetition)
        prob_dict = entry.get("predicted_probabilities", {})
        lookup[identity] = [float(prob_dict.get(name, 0.0)) for name in EMOTION_CLASSES]
    return lookup


def calculate_pair_baseline_relative_tv(
    video_results: List[VideoEvaluationResult],
    baseline_lookup: Dict[ClipIdentity, List[float]],
) -> Tuple[float, int]:
    """Baseline-relative TV deviation, averaged over same-actor pairs.
    Pairs whose baseline counterparts are missing in the lookup are skipped."""
    groups: Dict[str, List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = result.metadata.actor
        groups.setdefault(group_key, []).append(result)

    total_deviation = 0.0
    pair_count = 0
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            base_i = baseline_lookup.get(_clip_identity(group_results[i]))
            if base_i is None:
                continue
            p_pi = np.asarray(group_results[i].predicted_probabilities, dtype=np.float64)
            p_bi = np.asarray(base_i, dtype=np.float64)
            for j in range(i + 1, len(group_results)):
                base_j = baseline_lookup.get(_clip_identity(group_results[j]))
                if base_j is None:
                    continue
                p_pj = np.asarray(group_results[j].predicted_probabilities, dtype=np.float64)
                p_bj = np.asarray(base_j, dtype=np.float64)
                total_deviation += float(0.5 * np.sum(np.abs((p_pj - p_pi) - (p_bj - p_bi))))
                pair_count += 1

    return (total_deviation / pair_count if pair_count else 0.0), pair_count


def calculate_same_key_pair_baseline_relative_tv(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
    baseline_lookup: Dict[ClipIdentity, List[float]],
) -> Tuple[float, int]:
    """Within-key version, averaged across keys (weighted by pair count)."""
    total_deviation = 0.0
    total_pairs = 0
    for key_results in results_by_key.values():
        avg_dev, pair_count = calculate_pair_baseline_relative_tv(key_results, baseline_lookup)
        total_deviation += avg_dev * pair_count
        total_pairs += pair_count
    return (total_deviation / total_pairs if total_pairs else 0.0), total_pairs


def calculate_different_key_pair_baseline_relative_tv(
    video_results: List[VideoEvaluationResult],
    baseline_lookup: Dict[ClipIdentity, List[float]],
) -> Tuple[float, int]:
    """Cross-key version. Pair selection rule matches
    calculate_different_key_pair_consistency_tv (different key labels,
    both non-None). Baseline counterparts are matched by clip identity,
    independent of key, since the baseline has no keys."""
    groups: Dict[Tuple[str, str, str], List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = (result.metadata.actor, result.ground_truth_emotion, result.metadata.intensity)
        groups.setdefault(group_key, []).append(result)

    total_deviation = 0.0
    pair_count = 0
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            key_i = group_results[i].key_label
            if key_i is None:
                continue
            base_i = baseline_lookup.get(_clip_identity(group_results[i]))
            if base_i is None:
                continue
            p_pi = np.asarray(group_results[i].predicted_probabilities, dtype=np.float64)
            p_bi = np.asarray(base_i, dtype=np.float64)
            for j in range(i + 1, len(group_results)):
                key_j = group_results[j].key_label
                if key_j is None or key_i == key_j:
                    continue
                base_j = baseline_lookup.get(_clip_identity(group_results[j]))
                if base_j is None:
                    continue
                p_pj = np.asarray(group_results[j].predicted_probabilities, dtype=np.float64)
                p_bj = np.asarray(base_j, dtype=np.float64)
                total_deviation += float(0.5 * np.sum(np.abs((p_pj - p_pi) - (p_bj - p_bi))))
                pair_count += 1

    return (total_deviation / pair_count if pair_count else 0.0), pair_count


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
    groups: Dict[str, List[VideoEvaluationResult]] = {}
    for result in results:
        group_key = result.metadata.actor
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
    groups: Dict[str, List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = result.metadata.actor
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

def calculate_pair_consistency_tv(
    video_results: List[VideoEvaluationResult],
    by_group: Optional[List[str]] = None,
) -> Tuple[float, int]:
    """Pair consistency using TV distance over the full predicted distribution."""
    results = [r for r in video_results if by_group is None or r.filepath in by_group]
    groups: Dict[str, List[VideoEvaluationResult]] = {}
    for result in results:
        group_key = result.metadata.actor
        groups.setdefault(group_key, []).append(result)

    total_diff = 0.0
    pair_count = 0
    for group_results in groups.values():
        if len(group_results) < 2:
            continue
        for i in range(len(group_results)):
            for j in range(i + 1, len(group_results)):
                total_diff += total_variation_distance(
                    group_results[i].predicted_probabilities,
                    group_results[j].predicted_probabilities,
                )
                pair_count += 1

    return (total_diff / pair_count if pair_count else 0.0), pair_count


def calculate_same_key_pair_consistency_tv(
    results_by_key: Dict[str, List[VideoEvaluationResult]],
) -> Tuple[float, int]:
    total_diff = 0.0
    total_pairs = 0
    for key_results in results_by_key.values():
        avg_diff, pair_count = calculate_pair_consistency_tv(key_results)
        total_diff += avg_diff * pair_count
        total_pairs += pair_count
    return (total_diff / total_pairs if total_pairs else 0.0), total_pairs


def calculate_different_key_pair_consistency_tv(
    video_results: List[VideoEvaluationResult],
) -> Tuple[float, int]:
    groups: Dict[str, List[VideoEvaluationResult]] = {}
    for result in video_results:
        group_key = result.metadata.actor
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
            for j in range(i + 1, len(group_results)):
                key_j = group_results[j].key_label
                if key_j is None or key_i == key_j:
                    continue
                total_diff += total_variation_distance(
                    group_results[i].predicted_probabilities,
                    group_results[j].predicted_probabilities,
                )
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
