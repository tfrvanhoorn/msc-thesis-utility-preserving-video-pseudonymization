from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..data import iter_samples
from ..data.io import load_image
from ..pipeline import IdentitySeedPipeline
from ..utils.metrics import summarize_metrics
from ..components import Quantizer


logger = logging.getLogger(__name__)

DEBUG_DUMP_LIMIT = 3


class _DebugDumper:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._count = 0

    def should_dump(self) -> bool:
        if self.limit is None:
            return True
        with self._lock:
            if self._count >= self.limit:
                return False
            self._count += 1
            return True


_debug_dumper = _DebugDumper(limit=DEBUG_DUMP_LIMIT)


@dataclass
class SampleSummary:
    identity: str
    file_name: str
    seeds: List[str]
    detections: int
    embeddings: np.ndarray
    quantized: np.ndarray


def run_quantization_sweep(
    config: ExperimentConfig,
    pipeline: IdentitySeedPipeline,
    max_identities: int | None = None,
    max_per_identity: int | None = None,
    max_samples: int | None = None,
    batch_size: int = 1,
    num_workers: int = 1,
    pipeline_factory: "Callable[[], IdentitySeedPipeline]" | None = None,
) -> Dict[str, float]:
    sample_list = list(
        _iter_filtered_samples(
            config.data,
            max_identities=max_identities,
            max_per_identity=max_per_identity,
            max_samples=max_samples,
        )
    )
    sample_list, shared_state = _maybe_train_quantizer(config, pipeline, sample_list)
    if pipeline_factory and shared_state:
        base_factory = pipeline_factory

        def pipeline_factory(config=config, state=shared_state) -> IdentitySeedPipeline:
            fresh = base_factory()
            if hasattr(fresh.quantizer, "load_state"):
                fresh.quantizer.load_state(state)
            return fresh

    sample_stream = iter(sample_list)
    state = _SweepState()
    batch_size = max(batch_size, 1)
    if num_workers <= 1:
        _run_sequential(sample_stream, pipeline, batch_size, state)
    else:
        pipelines = _build_pipeline_pool(pipeline, num_workers, pipeline_factory)
        _run_parallel(sample_stream, pipelines, batch_size, state)
    return summarize_metrics(
        state.seeds_by_identity,
        embeddings_by_identity=state.embeddings_by_identity,
        quantized_by_identity=state.quantized_by_identity,
        quantizer=pipeline.quantizer,
    )


def _run_sequential(
    samples: Iterable[Any],
    pipeline: IdentitySeedPipeline,
    batch_size: int,
    state: "_SweepState",
) -> None:
    for batch in _batched(samples, batch_size):
        for sample in batch:
            image = load_image(sample.path)
            result = pipeline.process_image(image)
            _log_pipeline_debug(sample, result)
            summary = _summarize_sample(sample, result)
            state.record(summary)


def _run_parallel(
    samples: Iterable[Any],
    pipelines: Sequence[IdentitySeedPipeline],
    batch_size: int,
    state: "_SweepState",
) -> None:
    pipeline_pool = SimpleQueue()
    for pipe in pipelines:
        pipeline_pool.put(pipe)

    def worker(batch: Sequence[Any]) -> List[SampleSummary]:
        pipeline = pipeline_pool.get()
        try:
            summaries: List[SampleSummary] = []
            for sample in batch:
                image = load_image(sample.path)
                result = pipeline.process_image(image)
                _log_pipeline_debug(sample, result)
                summaries.append(_summarize_sample(sample, result))
            return summaries
        finally:
            pipeline_pool.put(pipeline)

    pending: set[Future[List[SampleSummary]]] = set()
    executor = ThreadPoolExecutor(max_workers=len(pipelines))
    try:
        for batch in _batched(samples, batch_size):
            if not batch:
                continue
            future = executor.submit(worker, batch)
            pending.add(future)
            if len(pending) >= len(pipelines) * 2:
                _drain_completed(pending, state)
        while pending:
            _drain_completed(pending, state)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received; cancelling %s batches", len(pending))
        _cancel_pending(pending)
        raise
    finally:
        _cancel_pending(pending)
        executor.shutdown(wait=False, cancel_futures=True)


def _drain_completed(pending, state: "_SweepState") -> None:
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        summaries = future.result()
        for summary in summaries:
            state.record(summary)
        pending.remove(future)


def _cancel_pending(pending: set[Future]) -> None:
    for future in list(pending):
        future.cancel()
        pending.remove(future)


def _summarize_sample(sample, result) -> SampleSummary:
    return SampleSummary(
        identity=sample.identity,
        file_name=sample.path.name,
        seeds=list(result.seeds),
        detections=len(result.detections),
        embeddings=result.embeddings,
        quantized=result.quantized,
    )


def _iter_filtered_samples(
    data_config,
    max_identities: int | None,
    max_per_identity: int | None,
    max_samples: int | None,
) -> Iterator[Any]:
    per_identity_counter: Dict[str, int] = defaultdict(int)
    seen_identities: set[str] = set()
    yielded = 0
    for sample in iter_samples(data_config):
        identity = sample.identity
        is_new_identity = identity not in seen_identities
        if max_identities is not None and is_new_identity and len(seen_identities) >= max_identities:
            continue
        if max_per_identity is not None and per_identity_counter[identity] >= max_per_identity:
            continue
        yield sample
        per_identity_counter[identity] += 1
        if is_new_identity:
            seen_identities.add(identity)
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def _batched(iterator: Iterable[Any], size: int) -> Iterator[Sequence[Any]]:
    batch: List[Any] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _maybe_train_quantizer(
    config: ExperimentConfig,
    pipeline: IdentitySeedPipeline,
    samples: Sequence[Any],
) -> tuple[Sequence[Any], dict[str, Any] | None]:
    qcfg = config.quantization
    train_split = getattr(qcfg, "train_split", 0.0) or 0.0
    max_train = getattr(qcfg, "max_train_samples", None)
    if train_split <= 0.0 or not samples:
        return samples, None

    rng_seed = getattr(qcfg, "random_seed", None)
    rng = np.random.default_rng(rng_seed)

    train_samples, eval_samples, train_ids, eval_ids = _grouped_train_eval_split(samples, train_split, rng)
    if not train_samples:
        return samples, None

    quantizer = pipeline.quantizer
    supports_training = quantizer.__class__.fit is not Quantizer.fit
    if not supports_training:
        logger.info("train_split provided but quantizer does not support training; skipping fit phase")
        return samples, None

    logger.info(
        "Training quantizer (%s) on %s samples from %s identities (train_split=%.2f, max_train_samples=%s); eval identities=%s",
        quantizer.__class__.__name__,
        len(train_samples),
        len(train_ids),
        train_split,
        max_train if max_train is not None else "none",
        len(eval_ids),
    )
    embeddings = _collect_embeddings_for_training(train_samples, pipeline, max_train)
    quantizer.fit(embeddings)
    if hasattr(quantizer, "_prototypes"):
        prot = getattr(quantizer, "_prototypes", None)
        if prot is not None:
            if isinstance(prot, list):
                shapes = [p.shape for p in prot if hasattr(p, "shape")]
                logger.info("Quantizer prototypes ready: subspaces=%s shapes=%s", len(prot), shapes)
            elif hasattr(prot, "shape"):
                logger.info("Quantizer prototypes ready: shape=%s", prot.shape)
    shared_state = None
    if hasattr(quantizer, "export_state"):
        shared_state = quantizer.export_state()
    return eval_samples if eval_samples else train_samples, shared_state


def _grouped_train_eval_split(
    samples: Sequence[Any],
    train_split: float,
    rng: np.random.Generator,
) -> tuple[list[Any], list[Any], list[str], list[str]]:
    by_id: dict[str, list[Any]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        identity = getattr(sample, "identity", None)
        if identity is None:
            identity = "__no_identity__"
        by_id[identity].append(sample)

    identities = list(by_id.keys())
    rng.shuffle(identities)
    total_samples = len(samples)
    target = int(total_samples * train_split)
    if train_split > 0.0 and target == 0:
        target = 1

    train_ids: list[str] = []
    train_count = 0
    for identity in identities:
        if target and train_count >= target:
            break
        train_ids.append(identity)
        train_count += len(by_id[identity])

    train_set = set(train_ids)
    train_samples = [s for identity in train_ids for s in by_id[identity]]
    eval_samples = [s for identity in identities if identity not in train_set for s in by_id[identity]]

    if not eval_samples and len(identities) > len(train_ids):
        remaining_ids = [i for i in identities if i not in train_set]
        if remaining_ids:
            move_id = remaining_ids[0]
            train_samples = [s for s in train_samples if getattr(s, "identity", None) != move_id]
            eval_samples.extend(by_id[move_id])
    return train_samples, eval_samples, train_ids, [i for i in identities if i not in train_set]


def _collect_embeddings_for_training(
    samples: Sequence[Any],
    pipeline: IdentitySeedPipeline,
    max_train_samples: int | None,
) -> np.ndarray:
    collected: list[np.ndarray] = []
    total = 0
    for idx, sample in enumerate(samples, start=1):
        image = load_image(sample.path)
        detections = pipeline.detector.detect(image)
        if not detections:
            continue
        aligned = [pipeline.aligner.align(image, det) for det in detections]
        embeddings = pipeline.embedder.embed(aligned)
        if embeddings.size == 0:
            continue
        collected.append(embeddings)
        total += embeddings.shape[0]
        if max_train_samples is not None and total >= max_train_samples:
            break
    if not collected:
        raise ValueError("No embeddings collected to train the quantizer. Adjust train_split or data filters.")
    stacked = np.vstack(collected)
    if max_train_samples is not None and stacked.shape[0] > max_train_samples:
        stacked = stacked[:max_train_samples]
    logger.info("Collected %s training embeddings for quantizer fit", stacked.shape[0])
    return stacked


def _build_pipeline_pool(
    base_pipeline: IdentitySeedPipeline,
    num_workers: int,
    factory: "Callable[[], IdentitySeedPipeline]" | None,
) -> List[IdentitySeedPipeline]:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    pipelines = [base_pipeline]
    if num_workers == 1:
        return pipelines
    if factory is None:
        raise ValueError("pipeline_factory must be provided when using multiple workers")
    for _ in range(num_workers - 1):
        pipelines.append(factory())
    return pipelines


class _SweepState:
    def __init__(self) -> None:
        self.seeds_by_identity: Dict[str, List[str]] = defaultdict(list)
        self.embeddings_by_identity: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.quantized_by_identity: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.processed_samples: int = 0

    def record(self, summary: SampleSummary) -> None:
        if summary.seeds:
            self.seeds_by_identity[summary.identity].extend(summary.seeds)
        if summary.embeddings is not None and getattr(summary.embeddings, "size", 0) > 0:
            self.embeddings_by_identity[summary.identity].append(summary.embeddings)
        if summary.quantized is not None and getattr(summary.quantized, "size", 0) > 0:
            self.quantized_by_identity[summary.identity].append(summary.quantized)
        self.processed_samples += 1
        status = f"seed_count={len(summary.seeds)}"
        if not summary.seeds:
            if summary.detections == 0:
                status += " (No faces detected)"
            else:
                status += " (Faces detected but no seeds generated)"
        logger.info(
            "Processed sample %s: identity=%s file=%s %s",
            self.processed_samples,
            summary.identity,
            summary.file_name,
            status,
        )


def _log_pipeline_debug(sample, result) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    emb_preview = _preview_array(result.embeddings)
    quant_preview = _preview_array(result.quantized)
    seed_preview = list(result.seeds[:5])
    logger.debug(
        "Debug sample identity=%s file=%s detections=%s seeds=%s\n  embeddings=%s\n  quantized=%s",
        sample.identity,
        sample.path.name,
        len(result.detections),
        seed_preview,
        emb_preview,
        quant_preview,
    )
    if _debug_dumper.should_dump():
        logger.debug(
            "Full embeddings (shape=%s):\n%s",
            getattr(result.embeddings, "shape", None),
            _format_full_array(result.embeddings),
        )
        logger.debug(
            "Full quantized vectors (shape=%s):\n%s",
            getattr(result.quantized, "shape", None),
            _format_full_array(result.quantized),
        )


def _preview_array(array: np.ndarray, max_rows: int = 2, max_cols: int = 5) -> List[List[float]]:
    if array is None or array.size == 0:
        return []
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = min(max_rows, arr.shape[0])
    cols = min(max_cols, arr.shape[1])
    preview = arr[:rows, :cols]
    return np.round(preview, 4).tolist()


def _format_full_array(array: np.ndarray) -> str:
    if array is None or array.size == 0:
        return "[]"
    arr = np.asarray(array)
    with np.printoptions(precision=6, suppress=True, threshold=arr.size, linewidth=120):
        return np.array2string(arr, separator=", ")
