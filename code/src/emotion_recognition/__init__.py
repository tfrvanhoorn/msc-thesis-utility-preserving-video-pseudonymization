"""Emotion recognition package for RAVDESS evaluation."""

from .ravdess_utils import (
    RAVDESSMetadata,
    RAVDESS_EMOTION_MAP,
    EMOTION_CLASSES,
    EMOTION_TO_IDX,
    parse_ravdess_filename,
    collect_ravdess_videos,
    validate_repetition_pairs,
    group_videos_by_metadata,
    emotion_to_one_hot,
)
from .emotion_metrics import (
    VideoEvaluationResult,
    MetricsReport,
    calculate_classification_accuracy,
    calculate_brier_score,
    calculate_repetition_consistency,
    aggregate_by_actor,
    aggregate_by_emotion,
    aggregate_by_intensity,
    generate_metrics_report,
)
from .emotion_inference import EmotionInferenceEngine
