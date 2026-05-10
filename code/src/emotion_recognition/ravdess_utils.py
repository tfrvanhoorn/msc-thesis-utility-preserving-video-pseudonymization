"""
RAVDESS dataset utility functions.

Handles parsing and metadata extraction from RAVDESS video filenames.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

RAVDESS_EMOTION_MAP = {
    "01": "Neutral",
    "02": "Neutral",
    "03": "Happiness",
    "04": "Sadness",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}

EMOTION_CLASSES = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
EMOTION_TO_IDX = {emotion: index for index, emotion in enumerate(EMOTION_CLASSES)}


@dataclass
class RAVDESSMetadata:
    filename: str
    modality: str
    vocal_channel: str
    emotion_code: str
    emotion_label: str
    intensity: str
    statement: str
    repetition: str
    actor: str
    sample_id: Optional[int] = None

    @property
    def actor_gender(self) -> str:
        actor_num = int(self.actor)
        return "male" if actor_num % 2 == 1 else "female"

    @property
    def intensity_label(self) -> str:
        return "normal" if self.intensity == "01" else "strong"

    def __hash__(self):
        return hash((self.actor, self.emotion_code, self.intensity, self.repetition))


def parse_ravdess_filename(filename: str, prefix_template: str = "video_sample{n}_") -> Optional[RAVDESSMetadata]:
    name_without_ext = Path(filename).stem
    escaped_prefix = re.escape(prefix_template)
    if "\\{n\\}" in escaped_prefix:
        prefix_pattern = escaped_prefix.replace("\\{n\\}", r"(\d+)")
        has_sample = True
    else:
        prefix_pattern = escaped_prefix
        has_sample = False
    pattern = rf"^{prefix_pattern}(\d{{2}})-(\d{{2}})-(\d{{2}})-(\d{{2}})-(\d{{2}})-(\d{{2}})-(\d{{2}})$"
    match = re.match(pattern, name_without_ext)
    if not match:
        return None

    if has_sample:
        sample_id_str, modality, vocal, emotion, intensity, statement, repetition, actor = match.groups()
        sample_id = int(sample_id_str)
    else:
        modality, vocal, emotion, intensity, statement, repetition, actor = match.groups()
        sample_id = None

    if modality not in ["01", "02", "03"]:
        return None
    if vocal not in ["01", "02"]:
        return None
    if emotion not in RAVDESS_EMOTION_MAP:
        return None
    if emotion == "02":
        return None
    if intensity not in ["01", "02"]:
        return None
    if emotion == "01" and intensity == "02":
        return None
    if statement not in ["01", "02"]:
        return None
    if repetition not in ["01", "02"]:
        return None
    if not (1 <= int(actor) <= 24):
        return None

    return RAVDESSMetadata(
        filename=filename,
        modality=modality,
        vocal_channel=vocal,
        emotion_code=emotion,
        emotion_label=RAVDESS_EMOTION_MAP[emotion],
        intensity=intensity,
        statement=statement,
        repetition=repetition,
        actor=actor,
        sample_id=sample_id,
    )


def collect_ravdess_videos(
    video_dir: Path,
    prefix_template: str = "video_sample{n}_",
) -> Tuple[Dict[str, RAVDESSMetadata], List[str]]:
    video_dir = Path(video_dir)
    metadata = {}
    failed = []
    video_extensions = [".mp4"]

    for video_file in video_dir.iterdir():
        if not video_file.is_file() or video_file.suffix.lower() not in video_extensions:
            continue
        parsed = parse_ravdess_filename(video_file.name, prefix_template)
        if parsed is None:
            failed.append(video_file.name)
        else:
            metadata[str(video_file)] = parsed

    return metadata, failed


def collect_keyed_ravdess_videos(
    video_dir: Path,
    num_keys: int,
    prefix_template: str = "video_sample{n}_",
) -> Tuple[Dict[str, Dict[str, RAVDESSMetadata]], List[Dict[str, str]]]:
    video_dir = Path(video_dir)
    keyed_metadata: Dict[str, Dict[str, RAVDESSMetadata]] = {}
    failed: List[Dict[str, str]] = []

    for key_index in range(1, num_keys + 1):
        key_label = f"key{key_index}"
        key_dir = video_dir / key_label
        if not key_dir.exists() or not key_dir.is_dir():
            failed.append({"key": key_label, "filename": str(key_dir), "reason": "missing directory"})
            continue

        metadata_dict, failed_files = collect_ravdess_videos(key_dir, prefix_template)
        keyed_metadata[key_label] = metadata_dict
        for failed_file in failed_files:
            failed.append({"key": key_label, "filename": failed_file, "reason": "invalid filename"})

    return keyed_metadata, failed


def validate_repetition_pairs(metadata_dict: Dict[str, RAVDESSMetadata]) -> Dict[Tuple[str, str, str], Tuple[Optional[str], Optional[str]]]:
    groups = {}
    for filepath, meta in metadata_dict.items():
        key = (meta.actor, meta.emotion_code, meta.intensity)
        if key not in groups:
            groups[key] = {"01": None, "02": None}
        groups[key][meta.repetition] = filepath

    return {key: (reps["01"], reps["02"]) for key, reps in groups.items()}


def group_videos_by_metadata(metadata_dict: Dict[str, RAVDESSMetadata]) -> Dict[str, Dict[str, List[str]]]:
    by_actor = {}
    by_emotion = {}
    by_intensity = {}
    by_repetition = {}

    for filepath, meta in metadata_dict.items():
        by_actor.setdefault(meta.actor, []).append(filepath)
        by_emotion.setdefault(meta.emotion_label, []).append(filepath)
        by_intensity.setdefault(meta.intensity_label, []).append(filepath)
        by_repetition.setdefault(meta.repetition, []).append(filepath)

    return {
        "by_actor": by_actor,
        "by_emotion": by_emotion,
        "by_intensity": by_intensity,
        "by_repetition": by_repetition,
    }


def emotion_to_one_hot(emotion_label: str) -> List[float]:
    if emotion_label not in EMOTION_TO_IDX:
        raise ValueError(f"Unknown emotion label: {emotion_label}")
    one_hot = [0.0] * len(EMOTION_CLASSES)
    one_hot[EMOTION_TO_IDX[emotion_label]] = 1.0
    return one_hot
