from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar.metrics import MetricsAccumulator  # noqa: E402


def test_metrics_semantics_are_coherent() -> None:
    metrics = MetricsAccumulator(
        anonymization_threshold=0.7,
        synchronism_threshold=0.7,
        diversity_threshold=0.7,
        differentiation_threshold=0.7,
        compute_auc_eer=False,
    )

    real = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    virtual = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    valid = torch.tensor([True, True])
    metrics.update_anonymization(real, virtual, valid)

    key1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    key2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    metrics.update_diversity(key1, key2)

    embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    metrics.update_differentiation(embeds, labels)

    sync_embeds_src_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sync_embeds_src_b = torch.tensor([[0.0, 1.0]])
    metrics.add_synchronism_embeddings(7, sync_embeds_src_a, source_id="a")
    metrics.add_synchronism_embeddings(7, sync_embeds_src_b, source_id="b")
    metrics.update_geometric_utility(2.0, 4.0)
    metrics.update_geometric_utility(6.0, 8.0)
    metrics.update_geometric_utility(None, None)

    summary = metrics.finalize()

    assert summary["anonymization_success_rate"] == 0.5
    assert summary["diversity_success_rate"] == 0.5
    assert summary["differentiation_success_rate"] == 1.0
    assert summary["thresholds"]["diversity"] == 0.7
    assert summary["thresholds"]["differentiation"] == 0.7
    assert summary["geometric_utility"]["head_posture_error"] == 4.0
    assert summary["geometric_utility"]["facial_expression_error"] == 6.0
    assert summary["geometric_utility"]["counts"]["valid_pairs"] == 2
    assert summary["geometric_utility"]["counts"]["invalid_pairs"] == 1


def test_auc_eer_is_reported_when_enabled() -> None:
    metrics = MetricsAccumulator(
        anonymization_threshold=0.7,
        synchronism_threshold=0.7,
        diversity_threshold=0.7,
        differentiation_threshold=0.7,
        compute_auc_eer=True,
    )

    metrics.update_anonymization(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.8, 0.2]]),
        torch.tensor([True, True]),
    )
    metrics.update_diversity(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    metrics.update_differentiation(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]]),
        torch.tensor([0, 1, 2]),
    )

    metrics.add_synchronism_embeddings(1, torch.tensor([[1.0, 0.0], [1.0, 0.0]]), source_id="src1")
    metrics.add_synchronism_embeddings(1, torch.tensor([[0.8, 0.2], [0.9, 0.1]]), source_id="src2")

    summary = metrics.finalize()

    for metric_name in [
        "anonymization",
        "synchronism_total",
        "synchronism_within",
        "synchronism_cross",
        "diversity",
        "differentiation",
    ]:
        metric = summary[metric_name]
        assert metric["auc"] is not None
        assert metric["eer"] is not None
        assert metric["eer_threshold"] is not None
