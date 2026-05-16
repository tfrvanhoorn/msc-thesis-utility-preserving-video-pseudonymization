#!/usr/bin/env python3
"""Plot validation loss learning curves across training runs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# When executed as a script, the utils folder may be added to sys.path and can
# shadow the stdlib logging module. Remove the script directory before imports.
_utils_dir = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry).resolve() != _utils_dir
]

if "logging" in sys.modules and not hasattr(sys.modules["logging"], "getLogger"):
    del sys.modules["logging"]

logging = importlib.import_module("logging")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

JSON_GLOB = "*_kfaar_projector_epoch_*.json"


@dataclass(frozen=True)
class LossSeries:
    label: str
    epochs: List[int]
    values: List[float]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_loss_series(input_dir: Path, label: str) -> Tuple[LossSeries, LossSeries]:
    json_paths = sorted(input_dir.glob(JSON_GLOB))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files matching {JSON_GLOB} in {input_dir}")

    w_reg_by_epoch: dict[int, float] = {}
    sum_by_epoch: dict[int, float] = {}

    for json_path in json_paths:
        payload = _load_json(json_path)
        epoch = payload.get("epoch")
        if epoch is None:
            logger.warning("Skipping %s: missing epoch", json_path)
            continue
        try:
            epoch_int = int(epoch)
        except (TypeError, ValueError):
            logger.warning("Skipping %s: invalid epoch %r", json_path, epoch)
            continue

        val_components = payload.get("val_loss_components")
        if not isinstance(val_components, dict):
            logger.warning("Skipping %s: missing val_loss_components", json_path)
            continue

        try:
            w_reg = float(val_components["w_reg"])
            sum_value = (
                float(val_components["ano"])
                + float(val_components["syn"])
                + float(val_components["div"])
                + float(val_components["dif"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping %s: invalid loss components (%s)", json_path, exc)
            continue

        w_reg_by_epoch[epoch_int] = w_reg
        sum_by_epoch[epoch_int] = sum_value

    if not w_reg_by_epoch:
        raise ValueError(f"No valid validation loss data found in {input_dir}")

    epochs_sorted = sorted(w_reg_by_epoch.keys())
    w_reg_values = [w_reg_by_epoch[epoch] for epoch in epochs_sorted]
    sum_values = [sum_by_epoch[epoch] for epoch in epochs_sorted]

    return (
        LossSeries(label=label, epochs=epochs_sorted, values=w_reg_values),
        LossSeries(label=label, epochs=epochs_sorted, values=sum_values),
    )


def _set_loss_axis(ax: plt.Axes, values: Iterable[float]) -> None:
    max_value = max(values, default=0.0)
    upper = max_value * 1.05 if max_value > 0 else 1.0
    ax.set_ylim(0.0, upper)


def _plot_series(
    series_list: List[LossSeries],
    y_label: str,
    save_path: Path,
    output_format: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    palette = plt.get_cmap("tab10")

    for idx, series in enumerate(series_list):
        color = palette(idx % 10)
        ax.plot(
            series.epochs,
            series.values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=series.label,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=True)

    all_values = [value for series in series_list for value in series.values]
    _set_loss_axis(ax, all_values)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{output_format}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation loss learning curves across training runs"
    )
    parser.add_argument(
        "--input_dir",
        action="append",
        required=True,
        help="Training output directory containing JSON checkpoint metadata",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for the corresponding input_dir (same order)",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to save learning curve plots",
    )
    parser.add_argument(
        "--output_format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for saved plots",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dirs = [Path(item) for item in args.input_dir]
    labels = args.label

    if len(input_dirs) != len(labels):
        raise ValueError("--input_dir and --label must be provided the same number of times")

    w_reg_series_list: List[LossSeries] = []
    sum_series_list: List[LossSeries] = []

    for input_dir, label in zip(input_dirs, labels):
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        w_reg_series, sum_series = _collect_loss_series(input_dir, label)
        w_reg_series_list.append(w_reg_series)
        sum_series_list.append(sum_series)

    save_dir = Path(args.save_dir)
    w_reg_path = save_dir / "loss_curve_w_reg"
    sum_path = save_dir / "loss_curve_sum_ano_syn_div_dif"

    _plot_series(
        w_reg_series_list,
        y_label="Validation W-Reg Loss",
        save_path=w_reg_path,
        output_format=args.output_format,
    )
    _plot_series(
        sum_series_list,
        y_label="Validation Anon+Syn+Div+Dif Loss",
        save_path=sum_path,
        output_format=args.output_format,
    )

    logger.info("Saved plots to %s", save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
