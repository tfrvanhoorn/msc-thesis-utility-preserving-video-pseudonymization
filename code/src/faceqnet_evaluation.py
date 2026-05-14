from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[0]
PROJECT_ROOT = current_file.parents[1]
EXTERNAL_LIB_ROOT = PROJECT_ROOT / "external_libraries"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXTERNAL_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_LIB_ROOT))

from data.prepared import (  # noqa: E402
    DEFAULT_PREPARED_REGEX,
    PreparedNameError,
    collect_prepared_videos,
    compile_prepared_regex,
    map_prepared_videos_by_key,
)
from faceqnet_metrics import FaceQnetEvaluator  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

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


def _sorted_key_names(keys: list[str]) -> list[str]:
    def _key_rank(name: str) -> tuple[int, str]:
        if name.startswith("key"):
            suffix = name[3:]
            if suffix.isdigit():
                return int(suffix), name
        return 10**9, name

    return sorted(keys, key=_key_rank)


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


def _load_video_frames(
    path: Path,
    *,
    max_frames: int | None = None,
    frame_step: int = 1,
    convert_rgb: bool = True,
) -> np.ndarray | None:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logging.warning("Failed to open video: %s", path)
        return None

    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_index % frame_step == 0:
                if convert_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        return None
    return np.stack(frames, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FaceQnet realism scores from prepared input/inferred folders")

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
        help="Directory to save FaceQnet evaluation reports",
    )
    parser.add_argument("--num_keys", type=int, default=None, help="Required key count; defaults to inferred from files")
    parser.add_argument("--detection_key", type=int, default=1, help="Key index used for FaceQnet evaluation when nested keys are used")

    parser.add_argument("--max_identities", type=int, default=None, help="Optional cap on number of identities to evaluate")
    parser.add_argument("--max_videos_per_identity", type=int, default=None, help="Optional cap on number of videos per identity to evaluate")

    parser.add_argument(
        "--faceqnet_model_path",
        type=Path,
        required=True,
        help="Path to FaceQnet .h5 model file",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device label to pass to FaceQnet evaluator")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.max_identities is not None and args.max_identities <= 0:
        raise ValueError("--max_identities must be > 0 when provided")
    if args.max_videos_per_identity is not None and args.max_videos_per_identity <= 0:
        raise ValueError("--max_videos_per_identity must be > 0 when provided")

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

    entries = _apply_entry_caps(
        entries,
        max_identities=args.max_identities,
        max_videos_per_identity=args.max_videos_per_identity,
    )

    key_names = [f"key{i}" for i in range(1, num_keys + 1)]
    if args.inferred_nested_keys:
        required_key = f"key{args.detection_key}"
        if required_key not in key_names:
            raise ValueError(f"--detection_key must be in [1, {num_keys}]")
    else:
        required_key = "key1"

    _log_pipe(
        "faceqnet_eval_start",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        num_entries=len(entries),
        num_keys=num_keys,
        detection_key=required_key,
        faceqnet_model_path=str(args.faceqnet_model_path),
    )

    evaluator = FaceQnetEvaluator(model_path=args.faceqnet_model_path, device=args.device)

    score_sum = 0.0
    valid_pairs = 0
    invalid_pairs = 0
    total_samples = 0
    identity_set: set[str] = set()

    batch_processing_start_time = time.perf_counter()

    with tqdm(total=len(entries), desc="Evaluating FaceQnet", unit="entry", dynamic_ncols=True) as progress:
        for entry in entries:
            progress.update(1)
            total_samples += 1
            identity_set.add(entry.identity)

            output_path = entry.outputs.get(required_key)
            if output_path is None:
                raise FileNotFoundError(
                    f"Entry {entry.input_video} is missing required output: {required_key}"
                )

            frames = _load_video_frames(output_path, max_frames=None, frame_step=1, convert_rgb=True)
            if frames is None:
                logging.warning("No frames found for %s", output_path)
                continue

            frame_count = int(frames.shape[0])
            for frame_idx in range(frame_count):
                score = evaluator.score_frame(frames[frame_idx])
                if score is None:
                    invalid_pairs += 1
                else:
                    score_sum += float(score)
                    valid_pairs += 1

    batch_processing_end_time = time.perf_counter()
    batch_processing_seconds = max(0.0, batch_processing_end_time - batch_processing_start_time)

    faceqnet_score = score_sum / float(valid_pairs) if valid_pairs else None

    metrics = {
        "faceqnet_score": faceqnet_score,
        "faceqnet_utility": {
            "faceqnet_score": faceqnet_score,
            "counts": {
                "faceqnet_valid_pairs": int(valid_pairs),
                "faceqnet_invalid_pairs": int(invalid_pairs),
            },
        },
        "counts": {
            "faceqnet_pairs_valid": int(valid_pairs),
            "faceqnet_pairs_invalid": int(invalid_pairs),
        },
    }

    _log_pipe(
        "faceqnet_eval_summary",
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        total_samples=total_samples,
        batch_processing_seconds=batch_processing_seconds,
        faceqnet_score=faceqnet_score,
        valid_pairs=valid_pairs,
        invalid_pairs=invalid_pairs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "faceqnet_eval_report.json"
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "inferred_dir": str(args.inferred_dir),
                "inferred_nested_keys": bool(args.inferred_nested_keys),
                "num_keys": num_keys,
                "enabled_metrics": ["faceqnet"],
                "metrics": metrics,
                "total_samples": total_samples,
                "timing": {
                    "batch_processing_seconds": batch_processing_seconds,
                },
                "settings": serialized_args,
                "identities": sorted(identity_set),
            },
            f,
            indent=2,
        )

    _log_pipe(
        "faceqnet_report_saved",
        path=str(report_path),
        input_dir=str(args.input_dir),
        inferred_dir=str(args.inferred_dir),
        batch_processing_seconds=batch_processing_seconds,
    )


if __name__ == "__main__":
    main()
