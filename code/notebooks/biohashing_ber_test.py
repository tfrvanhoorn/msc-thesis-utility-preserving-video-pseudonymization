import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from anon_pipeline.components.quantizer import BioHashingQuantizer
from anon_pipeline.pipeline import build_identity_seed_pipeline
from anon_pipeline.utils.config_loader import build_config, load_config_payload
from anon_pipeline.data import iter_samples
from anon_pipeline.data.io import load_image


def unpack_bits(packed: np.ndarray, bitorder: str, output_dim: int) -> np.ndarray:
    bits = np.unpackbits(packed, axis=1, bitorder=bitorder)
    return bits[:, :output_dim]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match, got {a.shape} vs {b.shape}")
    return (a != b).sum(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BioHashing BER on real samples")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/exp01_quantization.yaml")
    parser.add_argument("--max-identities", type=int, default=100)
    parser.add_argument("--max-per-identity", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--embedding-method", type=str, default="arcface", help="arcface or semantic")
    parser.add_argument("--quant-method", type=str, default="biohash")
    parser.add_argument("--bitorder", type=str, choices=["big", "little"], default="big")
    parser.add_argument("--packbits", action="store_true", default=True)
    parser.add_argument("--no-packbits", dest="packbits", action="store_false")
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace):
    payload = load_config_payload(args.config)
    payload.setdefault("embedding", {})["method"] = args.embedding_method
    if args.embedding_method.lower().startswith("arcface"):
        payload["embedding"]["feature_selector"] = {"keep": []}
    payload.setdefault("quantization", {})["method"] = args.quant_method
    cfg = build_config(payload)
    pipeline = build_identity_seed_pipeline(cfg)
    # Ensure BioHashing uses requested packing/bitorder
    if isinstance(pipeline.quantizer, BioHashingQuantizer):
        pipeline.quantizer.packbits = bool(args.packbits)
        pipeline.quantizer.bitorder = args.bitorder
    return cfg, pipeline


def iter_real_samples(cfg, max_ids: int | None, max_per_id: int | None, max_samples: int | None):
    per_identity_counter = defaultdict(int)
    seen_identities: set[str] = set()
    yielded = 0
    for sample in iter_samples(cfg.data):
        identity = sample.identity
        is_new = identity not in seen_identities
        if max_ids is not None and is_new and len(seen_identities) >= max_ids:
            continue
        if max_per_id is not None and per_identity_counter[identity] >= max_per_id:
            continue
        yield sample
        per_identity_counter[identity] += 1
        if is_new:
            seen_identities.add(identity)
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def get_embedding_dim(pipeline, sample) -> int | None:
    image = load_image(sample.path)
    detections = pipeline.detector.detect(image)
    if not detections:
        return None
    aligned = [pipeline.aligner.align(image, det) for det in detections]
    embeddings = pipeline.embedder.embed(aligned, source_paths=[sample.path] * len(aligned))
    if embeddings.size == 0:
        return None
    return embeddings.shape[1]


def ensure_quantizer_matches_dim(pipeline, embed_dim: int) -> None:
    q = pipeline.quantizer
    if isinstance(q, BioHashingQuantizer) and q.input_dim != embed_dim:
        pipeline.quantizer = BioHashingQuantizer(
            input_dim=embed_dim,
            output_dim=q.output_dim,
            random_seed=q.random_seed,
            packbits=q.packbits,
            bitorder=q.bitorder,
        )


def collect_hashes(pipeline, samples):
    quantized_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    samples = list(samples)
    embed_dim = None
    if samples:
        embed_dim = get_embedding_dim(pipeline, samples[0])
        if embed_dim is not None:
            ensure_quantizer_matches_dim(pipeline, embed_dim)
    for sample in samples:
        image = load_image(sample.path)
        result = pipeline.process_image(image, source_path=sample.path)
        if result.quantized.size == 0:
            continue
        quantized_by_id[sample.identity].append(result.quantized)
    return quantized_by_id


def to_bit_matrix(arrays: list[np.ndarray], quantizer: BioHashingQuantizer) -> np.ndarray:
    stacked = np.vstack(arrays)
    if getattr(quantizer, "packbits", False):
        bits = unpack_bits(stacked, getattr(quantizer, "bitorder", "big"), getattr(quantizer, "output_dim", stacked.shape[1] * 8))
    else:
        bits = stacked
    return bits.astype(np.uint8, copy=False)


def compute_ber(quantized_by_id, quantizer):
    same_id_distances = []
    diff_id_distances = []
    for identity, hashes in quantized_by_id.items():
        if len(hashes) < 2:
            continue
        bits = to_bit_matrix(hashes, quantizer)
        anchor = bits[0:1]
        others = bits[1:]
        dists = hamming_distance(np.repeat(anchor, others.shape[0], axis=0), others)
        same_id_distances.extend(dists)

    ids = list(quantized_by_id.keys())
    if len(ids) > 1:
        first_bits = {i: to_bit_matrix(hashes, quantizer) for i, hashes in quantized_by_id.items()}
        for i_idx, i in enumerate(ids):
            for j_idx, j in enumerate(ids):
                if i_idx == j_idx:
                    continue
                a = first_bits[i][0:1]
                b = first_bits[j][0:1]
                diff_id_distances.append(hamming_distance(a, b)[0])

    bit_len = getattr(quantizer, "output_dim", None)
    if bit_len is None:
        bit_len = to_bit_matrix([next(iter(quantized_by_id.values()))[0]], quantizer).shape[1]

    same_id_arr = np.array(same_id_distances, dtype=np.float32) if same_id_distances else np.array([], dtype=np.float32)
    diff_id_arr = np.array(diff_id_distances, dtype=np.float32) if diff_id_distances else np.array([], dtype=np.float32)

    return {
        "bit_length": bit_len,
        "same_id_mean_ber": float(same_id_arr.mean() / bit_len) if same_id_arr.size else None,
        "same_id_std_ber": float(same_id_arr.std() / bit_len) if same_id_arr.size else None,
        "diff_id_mean_ber": float(diff_id_arr.mean() / bit_len) if diff_id_arr.size else None,
        "diff_id_std_ber": float(diff_id_arr.std() / bit_len) if diff_id_arr.size else None,
        "same_id_pairs": int(same_id_arr.size),
        "diff_id_pairs": int(diff_id_arr.size),
        "num_identities": len(quantized_by_id),
        "total_samples": sum(len(v) for v in quantized_by_id.values()),
    }


def main():
    args = parse_args()
    cfg, pipeline = load_pipeline(args)
    samples = list(
        iter_real_samples(
            cfg,
            max_ids=args.max_identities,
            max_per_id=args.max_per_identity,
            max_samples=args.max_samples,
        )
    )
    quantized_by_id = collect_hashes(pipeline, samples)
    results = compute_ber(quantized_by_id, pipeline.quantizer)

    print("BioHashing BER on real samples")
    print(f"config: {args.config}")
    print(f"identities: {results['num_identities']} samples: {results['total_samples']}")
    for k in ["bit_length", "same_id_mean_ber", "same_id_std_ber", "diff_id_mean_ber", "diff_id_std_ber", "same_id_pairs", "diff_id_pairs"]:
        print(f"{k}: {results[k]}")


if __name__ == "__main__":
    main()
