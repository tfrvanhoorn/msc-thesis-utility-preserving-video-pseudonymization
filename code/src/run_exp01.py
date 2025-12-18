from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from anon_pipeline.experiments.exp01_quantization import run_quantization_sweep
from anon_pipeline.pipeline import IdentitySeedPipeline, build_identity_seed_pipeline
from anon_pipeline.utils.config_loader import build_config, load_config_payload
from anon_pipeline.utils.overrides import deep_merge


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 01 quantization sweeps.")
    parser.add_argument("--config", type=Path, default=Path("config/exp01_quantization.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/exp01_results.json"))
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--max-per-identity", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Global cap on processed samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Samples processed per worker invocation")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel pipelines (each loads detector/embedder)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Verbosity level: 0=warnings only, 1=info, 2=debug",
    )
    return parser.parse_args()


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def iter_sweeps(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sweeps = payload.get("sweeps") or [{"name": "default", "overrides": {}}]
    for idx, sweep in enumerate(sweeps):
        entry = {
            "name": sweep.get("name", f"run_{idx}"),
            "overrides": sweep.get("overrides", {}),
        }
        yield entry


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    base_payload = load_config_payload(args.config)

    runs = []
    for sweep in iter_sweeps(base_payload):
        merged = deep_merge(base_payload, sweep["overrides"])
        merged.pop("sweeps", None)
        config = build_config(merged)
        pipeline = build_identity_seed_pipeline(config)
        pipeline_factory = None
        if args.workers > 1:
            def pipeline_factory(config=config) -> IdentitySeedPipeline:
                return build_identity_seed_pipeline(config)
        metrics = run_quantization_sweep(
            config,
            pipeline,
            max_identities=args.max_identities,
            max_per_identity=args.max_per_identity,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pipeline_factory=pipeline_factory,
        )
        runs.append({
            "name": sweep["name"],
            "config": merged,
            "metrics": metrics,
        })

    for run in runs:
        metrics = run["metrics"]
        logger.info(
            "Run %s -> samples=%s identities=%s consistency=%.4f collision=%.4f",
            run["name"],
            metrics.get("total_samples"),
            metrics.get("num_identities"),
            metrics.get("consistency_rate", 0.0),
            metrics.get("collision_rate", 0.0),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "exp01_quantization",
                "timestamp": datetime.utcnow().isoformat(),
                "runs": runs,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
