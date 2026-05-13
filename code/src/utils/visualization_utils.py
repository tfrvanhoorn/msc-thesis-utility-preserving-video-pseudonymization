#!/usr/bin/env python3
"""Utilities to visualize emotion recognition findings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# When executed as a script, the utils folder is added to sys.path and can
# shadow the stdlib logging module. Remove it before importing logging/matplotlib.
if sys.path and Path(sys.path[0]).name == "utils":
    sys.path.pop(0)

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = [
    "Neutral",
    "Happiness",
    "Sadness",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
]


def _parse_emotions(raw: Optional[str]) -> List[str]:
    if raw is None:
        return DEFAULT_EMOTIONS
    emotions = [item.strip() for item in raw.split(",") if item.strip()]
    if not emotions:
        raise ValueError("--emotions must include at least one emotion label")
    return emotions


def _load_report(report_path: Path) -> Dict:
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_results_by_filename(report: Dict) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in report.get("video_results", []):
        filename = entry.get("filename")
        if filename is None:
            continue
        grouped.setdefault(filename, []).append(entry)
    return grouped


def _sort_entries_by_key(entries: List[Dict]) -> List[Dict]:
    def key_label(entry: Dict) -> str:
        raw = entry.get("key")
        return "" if raw is None else str(raw)

    return sorted(entries, key=key_label)


def _select_clip_entry(
    entries_by_filename: Dict[str, List[Dict]],
    filename: str,
    preferred_key: Optional[str],
    allow_different_key: bool,
) -> Tuple[Dict, Optional[str]]:
    if filename not in entries_by_filename:
        raise ValueError(f"Filename not found in report: {filename}")

    entries = _sort_entries_by_key(entries_by_filename[filename])
    if preferred_key is None:
        entry = entries[0]
        return entry, entry.get("key")

    if allow_different_key:
        for entry in entries:
            if entry.get("key") != preferred_key:
                return entry, entry.get("key")
        raise ValueError(
            f"No alternative key found for {filename}; available keys: "
            f"{[e.get('key') for e in entries]}"
        )

    for entry in entries:
        if entry.get("key") == preferred_key:
            return entry, entry.get("key")

    raise ValueError(
        f"Key {preferred_key} not found for {filename}; available keys: "
        f"{[e.get('key') for e in entries]}"
    )


def _resolve_emotion_values(entry: Dict, emotions: List[str]) -> List[float]:
    probabilities = entry.get("predicted_probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("Missing predicted_probabilities in report entry")
    values = []
    for emotion in emotions:
        if emotion not in probabilities:
            raise ValueError(f"Emotion {emotion} not found in predicted_probabilities")
        values.append(float(probabilities[emotion]))
    return values


def _build_color_map(emotions: List[str]) -> Dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {emotion: palette(idx % 10) for idx, emotion in enumerate(emotions)}


def plot_progress_line_chart(
    report_path: Path,
    clip1_name: str,
    clip2_name: str,
    emotions: List[str],
    save_dir: Path,
    use_different_key: bool,
    output_format: str,
) -> Path:
    report = _load_report(report_path)
    entries_by_filename = _group_results_by_filename(report)

    clip1_entry, clip1_key = _select_clip_entry(
        entries_by_filename,
        clip1_name,
        preferred_key=None,
        allow_different_key=False,
    )

    clip2_entry, clip2_key = _select_clip_entry(
        entries_by_filename,
        clip2_name,
        preferred_key=clip1_key,
        allow_different_key=use_different_key,
    )

    clip1_values = _resolve_emotion_values(clip1_entry, emotions)
    clip2_values = _resolve_emotion_values(clip2_entry, emotions)

    colors = _build_color_map(emotions)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    x_positions = [0, 1]
    x_labels = ["Clip 1", "Clip 2"]

    for emotion, v1, v2 in zip(emotions, clip1_values, clip2_values):
        ax.plot(
            x_positions,
            [v1, v2],
            marker="o",
            label=emotion,
            color=colors[emotion],
            linewidth=2.4,
            markersize=9,
        )
        delta = v2 - v1
        ax.text(
            1.02,
            v2,
            f"{delta:+.3f}",
            color=colors[emotion],
            va="center",
            fontsize=16,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    ax.set_xticks(x_positions, x_labels, fontsize=14)
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Confidence", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=13, frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    save_dir.mkdir(parents=True, exist_ok=True)
    safe_clip1 = Path(clip1_name).stem.replace(" ", "_")
    safe_clip2 = Path(clip2_name).stem.replace(" ", "_")
    key_mode = "different_key" if use_different_key else "same_key"
    output_path = save_dir / f"emotion_progress_{safe_clip1}_to_{safe_clip2}_{key_mode}.{output_format}"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot emotion confidence progress between two clips"
    )
    parser.add_argument("--report-json", required=True, help="Path to report JSON")
    parser.add_argument("--clip1", required=True, help="Filename of the first clip")
    parser.add_argument("--clip2", required=True, help="Filename of the second clip")
    parser.add_argument(
        "--emotions",
        default=None,
        help="Comma-separated list of emotions to plot",
    )
    parser.add_argument("--save-dir", required=True, help="Directory to save figures")
    parser.add_argument(
        "--output-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )

    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--same-key",
        action="store_true",
        help="Use the same key for clip2 if available",
    )
    key_group.add_argument(
        "--different-key",
        action="store_true",
        help="Use a different key for clip2 if available",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    report_path = Path(args.report_json)
    if not report_path.exists():
        logger.error("Report JSON not found: %s", report_path)
        return 1

    emotions = _parse_emotions(args.emotions)
    save_dir = Path(args.save_dir)
    use_different_key = bool(args.different_key)

    try:
        output_path = plot_progress_line_chart(
            report_path=report_path,
            clip1_name=args.clip1,
            clip2_name=args.clip2,
            emotions=emotions,
            save_dir=save_dir,
            use_different_key=use_different_key,
            output_format=args.output_format,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Saved plot to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
