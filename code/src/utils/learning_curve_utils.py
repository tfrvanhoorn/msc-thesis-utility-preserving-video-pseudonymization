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


@dataclass(frozen=True)
class RunLossData:
    label: str
    epochs: List[int]
    components: dict[str, List[float]]
    weighted_components: dict[str, List[float]]
    w_reg: List[float]
    weighted_w_reg: List[float]
    sum_components: List[float]
    weighted_sum_components: List[float]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_run_loss_data(input_dir: Path, label: str) -> RunLossData:
    json_paths = sorted(input_dir.glob(JSON_GLOB))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files matching {JSON_GLOB} in {input_dir}")

    components_by_epoch: dict[int, dict[str, float]] = {}
    weighted_by_epoch: dict[int, dict[str, float]] = {}
    w_reg_by_epoch: dict[int, float] = {}
    weighted_w_reg_by_epoch: dict[int, float] = {}
    sum_by_epoch: dict[int, float] = {}
    weighted_sum_by_epoch: dict[int, float] = {}

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

        config = payload.get("config")
        if not isinstance(config, dict):
            config = {}

        try:
            ano = float(val_components["ano"])
            syn = float(val_components["syn"])
            div = float(val_components["div"])
            dif = float(val_components["dif"])
            w_reg = float(val_components["w_reg"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping %s: invalid loss components (%s)", json_path, exc)
            continue

        try:
            lambda_ano = float(config.get("lambda_ano", 1.0))
            lambda_syn = float(config.get("lambda_syn", 1.0))
            lambda_div = float(config.get("lambda_div", 1.0))
            lambda_dif = float(config.get("lambda_dif", 1.0))
            lambda_w_reg = float(config.get("lambda_w_reg", 1.0))
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping %s: invalid lambda config (%s)", json_path, exc)
            continue

        components_by_epoch[epoch_int] = {
            "ano": ano,
            "syn": syn,
            "div": div,
            "dif": dif,
        }
        weighted_by_epoch[epoch_int] = {
            "ano": ano * lambda_ano,
            "syn": syn * lambda_syn,
            "div": div * lambda_div,
            "dif": dif * lambda_dif,
        }
        w_reg_by_epoch[epoch_int] = w_reg
        weighted_w_reg_by_epoch[epoch_int] = w_reg * lambda_w_reg
        sum_by_epoch[epoch_int] = ano + syn + div + dif
        weighted_sum_by_epoch[epoch_int] = (
            ano * lambda_ano + syn * lambda_syn + div * lambda_div + dif * lambda_dif
        )

    if not w_reg_by_epoch:
        raise ValueError(f"No valid validation loss data found in {input_dir}")

    epochs_sorted = sorted(w_reg_by_epoch.keys())
    components: dict[str, List[float]] = {"ano": [], "syn": [], "div": [], "dif": []}
    weighted_components: dict[str, List[float]] = {"ano": [], "syn": [], "div": [], "dif": []}

    for epoch in epochs_sorted:
        for key in components:
            components[key].append(components_by_epoch[epoch][key])
            weighted_components[key].append(weighted_by_epoch[epoch][key])

    return RunLossData(
        label=label,
        epochs=epochs_sorted,
        components=components,
        weighted_components=weighted_components,
        w_reg=[w_reg_by_epoch[epoch] for epoch in epochs_sorted],
        weighted_w_reg=[weighted_w_reg_by_epoch[epoch] for epoch in epochs_sorted],
        sum_components=[sum_by_epoch[epoch] for epoch in epochs_sorted],
        weighted_sum_components=[weighted_sum_by_epoch[epoch] for epoch in epochs_sorted],
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
    y_min_zero: bool,
    title: str | None = None,
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
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=True)

    if y_min_zero:
        all_values = [value for series in series_list for value in series.values]
        _set_loss_axis(ax, all_values)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{output_format}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_component_series(
    runs: List[RunLossData],
    components: List[str],
    y_label: str,
    save_path: Path,
    output_format: str,
    use_weighted: bool,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    palette = plt.get_cmap("tab10")

    line_styles = ["-", "--"]

    display_names = {
        "ano": "ano",
        "syn": "Con",
        "div": "div",
        "dif": "dif",
    }

    for run_idx, run in enumerate(runs):
        style = line_styles[0] if run_idx == 0 else line_styles[1]
        data = run.weighted_components if use_weighted else run.components
        for comp_idx, comp in enumerate(components):
            color = palette(comp_idx % 10)
            ax.plot(
                run.epochs,
                data[comp],
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=f"{run.label} | {display_names.get(comp, comp)}",
                color=color,
                linestyle=style,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=True, ncol=2)

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

    runs: List[RunLossData] = []

    for input_dir, label in zip(input_dirs, labels):
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        runs.append(_collect_run_loss_data(input_dir, label))

    save_dir = Path(args.save_dir)
    w_reg_path = save_dir / "loss_curve_w_reg"
    w_reg_weighted_path = save_dir / "loss_curve_w_reg_weighted"
    sum_path = save_dir / "loss_curve_sum_ano_syn_div_dif"
    sum_weighted_path = save_dir / "loss_curve_sum_ano_syn_div_dif_weighted"
    component_path = save_dir / "loss_curve_components"
    component_weighted_path = save_dir / "loss_curve_components_weighted"

    w_reg_series_list = [
        LossSeries(label=run.label, epochs=run.epochs, values=run.w_reg) for run in runs
    ]
    w_reg_weighted_series_list = [
        LossSeries(label=run.label, epochs=run.epochs, values=run.weighted_w_reg)
        for run in runs
    ]
    sum_series_list = [
        LossSeries(label=run.label, epochs=run.epochs, values=run.sum_components)
        for run in runs
    ]
    sum_weighted_series_list = [
        LossSeries(label=run.label, epochs=run.epochs, values=run.weighted_sum_components)
        for run in runs
    ]

    _plot_series(
        w_reg_series_list,
        y_label="Validation W-Reg Loss",
        save_path=w_reg_path,
        output_format=args.output_format,
        y_min_zero=True,
        title="Learning curve for Validation Regularization Loss over 10 epochs",
    )
    _plot_series(
        w_reg_weighted_series_list,
        y_label="Validation Weighted W-Reg Loss",
        save_path=w_reg_weighted_path,
        output_format=args.output_format,
        y_min_zero=True,
        title="Learning curve for Validation Weighted Regularization Loss over 10 epochs",
    )
    _plot_series(
        sum_series_list,
        y_label="Validation Anon+Con+Div+Dif Loss",
        save_path=sum_path,
        output_format=args.output_format,
        y_min_zero=False,
        title="Learning curve for Validation Combined Identity Loss over 10 epochs",
    )
    _plot_series(
        sum_weighted_series_list,
        y_label="Validation Weighted Anon+Con+Div+Dif Loss",
        save_path=sum_weighted_path,
        output_format=args.output_format,
        y_min_zero=False,
        title="Learning curve for Validation Weighted Combined Identity Loss over 10 epochs",
    )

    _plot_component_series(
        runs,
        components=["ano", "syn", "div", "dif"],
        y_label="Validation Component Loss",
        save_path=component_path,
        output_format=args.output_format,
        use_weighted=False,
        title="Learning curve for Validation Component Loss over 10 epochs",
    )
    _plot_component_series(
        runs,
        components=["ano", "syn", "div", "dif"],
        y_label="Validation Weighted Component Loss",
        save_path=component_weighted_path,
        output_format=args.output_format,
        use_weighted=True,
        title="Learning curve for Validation Weighted Component Loss over 10 epochs",
    )

    logger.info("Saved plots to %s", save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
