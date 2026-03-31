from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

current_file = Path(__file__).resolve()
SRC_ROOT = current_file.parents[1]
PROJECT_ROOT = current_file.parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.video_io import load_video_frames  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

DEFAULT_VIDEO_PATTERNS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
DEFAULT_MODEL_ID = "nateraw/videomae-base-finetuned-ucf101-subset"
DEFAULT_FAIL_ON_INFERENCE_ERROR = True
DEFAULT_UNKNOWN_LABELS_LOG_LIMIT = 25


@dataclass(frozen=True)
class VideoEntry:
    true_folder_label: str
    video_path: Path


def _log_pipe(category: str, **fields: Any) -> None:
    parts = [category]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logging.info(" | ".join(parts))


def _normalize_label(text: str) -> str:
    value = text.strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _serialize_arg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _iter_video_entries(input_dir: Path, extensions: tuple[str, ...], recursive: bool) -> list[VideoEntry]:
    label_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    entries: list[VideoEntry] = []
    for label_dir in label_dirs:
        for pattern in extensions:
            iterator = label_dir.rglob(pattern) if recursive else label_dir.glob(pattern)
            for path in iterator:
                if path.is_file():
                    entries.append(VideoEntry(true_folder_label=label_dir.name, video_path=path))
    entries = sorted(entries, key=lambda item: str(item.video_path).lower())
    return entries


def _build_label_index(id2label: dict[int, str]) -> dict[str, list[tuple[int, str]]]:
    by_normalized: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, label in id2label.items():
        by_normalized[_normalize_label(label)].append((index, label))
    return by_normalized


def _resolve_folder_label(
    folder_label: str,
    by_normalized: dict[str, list[tuple[int, str]]],
) -> tuple[int | None, str | None, str | None]:
    normalized = _normalize_label(folder_label)
    candidates = by_normalized.get(normalized, [])
    if not candidates:
        return None, None, "label_not_in_model"
    if len(candidates) == 1:
        idx, label = candidates[0]
        return idx, label, None

    by_casefold = [pair for pair in candidates if pair[1].casefold() == folder_label.casefold()]
    if len(by_casefold) == 1:
        idx, label = by_casefold[0]
        return idx, label, None

    return None, None, "ambiguous_normalized_label"


def _clip_starts(num_frames_total: int, clip_length: int) -> list[int]:
    if num_frames_total <= clip_length:
        return [0]
    starts = list(range(0, num_frames_total, clip_length))
    if starts[-1] != num_frames_total - clip_length:
        starts[-1] = min(starts[-1], num_frames_total - clip_length)
    return starts


def _predict_video(
    model: VideoMAEForVideoClassification,
    processor: AutoImageProcessor,
    frames_rgb: np.ndarray,
    clip_length: int,
    device: torch.device,
) -> dict[str, Any]:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected video frames of shape (T,H,W,3), got {tuple(frames_rgb.shape)}")

    total_frames = int(frames_rgb.shape[0])
    if total_frames <= 0:
        raise ValueError("Video contains no frames")

    logits_per_clip: list[torch.Tensor] = []
    starts = _clip_starts(total_frames, clip_length)

    with torch.no_grad():
        for start in starts:
            stop = min(start + clip_length, total_frames)
            clip = frames_rgb[start:stop]
            if clip.shape[0] < clip_length:
                pad = np.repeat(clip[-1:,...], clip_length - clip.shape[0], axis=0)
                clip = np.concatenate([clip, pad], axis=0)

            inputs = processor(list(clip), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            output = model(pixel_values=pixel_values)
            logits_per_clip.append(output.logits[0].detach().cpu())

    stacked = torch.stack(logits_per_clip, dim=0)
    mean_logits = stacked.mean(dim=0)
    probs = torch.softmax(mean_logits, dim=-1)
    pred_idx = int(torch.argmax(probs).item())
    pred_score = float(probs[pred_idx].item())

    topk = min(5, probs.numel())
    values, indices = torch.topk(probs, k=topk)
    topk_predictions = [
        {
            "label": model.config.id2label[int(i.item())],
            "score": float(v.item()),
        }
        for v, i in zip(values, indices)
    ]

    return {
        "predicted_label_index": pred_idx,
        "predicted_label": model.config.id2label[pred_idx],
        "confidence": pred_score,
        "num_frames": total_frames,
        "num_clips": len(logits_per_clip),
        "topk": topk_predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate action recognition accuracy on folder-labeled videos using "
            "VideoMAE UCF101 subset model"
        )
    )

    parser.add_argument("--input_dir", type=Path, required=True, help="Input root with first-level label folders")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where JSON report is saved")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--video_extensions",
        type=str,
        default=",".join(DEFAULT_VIDEO_PATTERNS),
        help="Comma-separated glob extensions to scan under each label folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan inside each label folder for videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap for number of discovered videos (after sorting)",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=None,
        help="Clip length for VideoMAE inference; defaults to model config num_frames",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="action_recognition_eval_report.json",
        help="JSON filename inside output_dir",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    fail_on_inference_error = DEFAULT_FAIL_ON_INFERENCE_ERROR
    unknown_labels_log_limit = DEFAULT_UNKNOWN_LABELS_LOG_LIMIT

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    extensions = tuple(part.strip() for part in args.video_extensions.split(",") if part.strip())
    if not extensions:
        raise ValueError("--video_extensions must provide at least one extension pattern")
    if unknown_labels_log_limit < 1:
        raise ValueError("DEFAULT_UNKNOWN_LABELS_LOG_LIMIT must be >= 1")

    device = torch.device(args.device)

    _log_pipe(
        "action_eval_start",
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        model_id=args.model_id,
        device=str(device),
    )

    model = VideoMAEForVideoClassification.from_pretrained(args.model_id)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = model.to(device)
    model.eval()

    clip_length = int(args.clip_length or getattr(model.config, "num_frames", 16))
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")

    id2label_raw = getattr(model.config, "id2label", None)
    if not isinstance(id2label_raw, dict) or not id2label_raw:
        raise RuntimeError("Model config does not expose id2label mapping")

    id2label: dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}
    label_index = _build_label_index(id2label)
    _log_pipe("action_eval_model_labels", num_labels=len(id2label))

    entries = _iter_video_entries(args.input_dir, extensions=extensions, recursive=bool(args.recursive))
    if args.max_videos is not None:
        entries = entries[: args.max_videos]

    _log_pipe("action_eval_discovery", discovered_videos=len(entries), clip_length=clip_length)

    counters: dict[str, int] = defaultdict(int)
    per_class_support: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    predicted_count_by_class: dict[str, int] = defaultdict(int)
    actual_count_by_class: dict[str, int] = defaultdict(int)
    unknown_label_folders: dict[str, int] = defaultdict(int)

    per_video_records: list[dict[str, Any]] = []
    start_ts = time.time()

    for entry in tqdm(entries, desc="Evaluating action recognition", unit="video"):
        counters["total_seen"] += 1

        true_idx, resolved_true_label, resolve_error = _resolve_folder_label(entry.true_folder_label, label_index)
        if true_idx is None or resolved_true_label is None:
            counters["skipped_unknown_label"] += 1
            unknown_label_folders[entry.true_folder_label] += 1
            per_video_records.append(
                {
                    "video_path": str(entry.video_path),
                    "true_folder_label": entry.true_folder_label,
                    "resolved_true_label": None,
                    "status": "skipped_unknown_label",
                    "error_reason": resolve_error,
                }
            )
            continue

        frames = load_video_frames(entry.video_path, convert_rgb=True)
        if frames is None or frames.size == 0:
            counters["skipped_unreadable"] += 1
            per_video_records.append(
                {
                    "video_path": str(entry.video_path),
                    "true_folder_label": entry.true_folder_label,
                    "resolved_true_label": resolved_true_label,
                    "status": "skipped_unreadable",
                    "error_reason": "failed_to_read_frames",
                }
            )
            continue

        try:
            prediction = _predict_video(
                model=model,
                processor=processor,
                frames_rgb=frames,
                clip_length=clip_length,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            counters["inference_errors"] += 1
            readable_error = f"{type(exc).__name__}: {exc}"
            per_video_records.append(
                {
                    "video_path": str(entry.video_path),
                    "true_folder_label": entry.true_folder_label,
                    "resolved_true_label": resolved_true_label,
                    "status": "inference_error",
                    "error_reason": readable_error,
                }
            )
            logging.error(
                "action_eval_inference_error | video_path=%s | true_folder_label=%s | resolved_true_label=%s | error=%s",
                str(entry.video_path),
                entry.true_folder_label,
                resolved_true_label,
                readable_error,
            )
            if fail_on_inference_error:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                raise RuntimeError(
                    "Inference failed and fail-on-inference-error default is enabled | "
                    f"video_path={entry.video_path} | true_folder_label={entry.true_folder_label} | "
                    f"resolved_true_label={resolved_true_label} | error={readable_error}\n{tb}"
                ) from exc
            continue

        counters["evaluated"] += 1
        is_correct = int(prediction["predicted_label_index"] == true_idx)
        counters["correct"] += is_correct
        counters["incorrect"] += 1 - is_correct

        actual_count_by_class[resolved_true_label] += 1
        predicted_count_by_class[prediction["predicted_label"]] += 1
        per_class_support[resolved_true_label] += 1
        per_class_correct[resolved_true_label] += is_correct

        per_video_records.append(
            {
                "video_path": str(entry.video_path),
                "true_folder_label": entry.true_folder_label,
                "resolved_true_label": resolved_true_label,
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "num_frames": prediction["num_frames"],
                "num_clips": prediction["num_clips"],
                "topk": prediction["topk"],
                "is_correct": bool(is_correct),
                "status": "evaluated",
                "error_reason": None,
            }
        )

    elapsed_seconds = float(time.time() - start_ts)

    evaluated = counters["evaluated"]
    overall_accuracy = float(counters["correct"] / evaluated) if evaluated > 0 else None

    per_class_metrics: dict[str, dict[str, Any]] = {}
    for label in sorted(per_class_support.keys()):
        support = per_class_support[label]
        correct = per_class_correct[label]
        per_class_metrics[label] = {
            "support": int(support),
            "correct": int(correct),
            "accuracy": float(correct / support) if support > 0 else None,
        }

    summary = {
        "total_seen": int(counters["total_seen"]),
        "evaluated": int(counters["evaluated"]),
        "correct": int(counters["correct"]),
        "incorrect": int(counters["incorrect"]),
        "accuracy": overall_accuracy,
        "skipped_unknown_label": int(counters["skipped_unknown_label"]),
        "skipped_unreadable": int(counters["skipped_unreadable"]),
        "inference_errors": int(counters["inference_errors"]),
        "elapsed_seconds": elapsed_seconds,
    }

    if unknown_label_folders:
        unknown_items = sorted(unknown_label_folders.items(), key=lambda item: (-item[1], item[0]))
        shown = unknown_items[: unknown_labels_log_limit]
        for label, count in shown:
            _log_pipe("action_eval_unknown_label", label=label, skipped_videos=int(count))
        hidden_count = max(0, len(unknown_items) - len(shown))
        _log_pipe(
            "action_eval_unknown_labels_summary",
            unique_unknown_labels=len(unknown_items),
            shown=len(shown),
            hidden=hidden_count,
            skipped_unknown_label_total=int(counters["skipped_unknown_label"]),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_filename

    serialized_args = {
        key: _serialize_arg_value(value)
        for key, value in vars(args).items()
    }

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": {
            "model_id": args.model_id,
            "clip_length": clip_length,
            "num_labels": len(id2label),
            "id2label": {str(k): v for k, v in sorted(id2label.items())},
        },
        "settings": serialized_args,
        "summary": summary,
        "per_class_metrics": per_class_metrics,
        "actual_count_by_class": {k: int(v) for k, v in sorted(actual_count_by_class.items())},
        "predicted_count_by_class": {k: int(v) for k, v in sorted(predicted_count_by_class.items())},
        "unknown_label_folders": {k: int(v) for k, v in sorted(unknown_label_folders.items())},
        "videos": per_video_records,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _log_pipe(
        "action_eval_complete",
        report_path=str(report_path),
        total_seen=summary["total_seen"],
        evaluated=summary["evaluated"],
        correct=summary["correct"],
        accuracy=summary["accuracy"],
        skipped_unknown_label=summary["skipped_unknown_label"],
        skipped_unreadable=summary["skipped_unreadable"],
        inference_errors=summary["inference_errors"],
    )


if __name__ == "__main__":
    main()
