# Code workspace

This area collects all runnable components, experiment definitions, and supporting utilities.

## Layout

```
code/
├── config/               # YAML files describing tunable parameters per pipeline/experiment
├── experiments/          # Markdown specs + notebooks or scripts per experiment
├── requirements.txt      # Python dependencies for the code workspace
└── src/
	└── anon_pipeline/
		├── components/   # Swappable algorithms (detector, embedder, quantizer, hasher)
		├── pipeline/     # High-level orchestration of the identity-seed workflow
		└── utils/        # Shared helpers (config loading, image ops, logging)
```

Components expose clear interfaces so that alternative models (e.g., swapping ArcFace for AdaFace) only requires updating the configuration file. Tunable parameters such as quantization grid size or detection thresholds are centralized under `config/` and read at runtime.

## Usage sketch

### Create & activate the virtual environment (Windows PowerShell)

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> ℹ️ The first run may download pretrained weights (buffalo_l detector, arcface_r100_v1 embedder) to `~/.insightface`. Specify `detector.root` / `embedding.root` in the config to control the cache directory.

### Run Experiment 01 (CelebA)

```powershell
cd code
.\.venv\Scripts\Activate.ps1  # if not already active
python -m src.run_exp01 --config config/exp01_quantization.yaml --output results/exp01.json
```

`config/exp01_quantization.yaml` is already configured to use the CelebA loader (`dataset_type: celeba` pointing at `data/celeba`). Duplicate it if you need alternative sweeps, detectors, or limits (`max_per_identity`, `max_samples`).

Runtime shortcuts:
- `--max-identities`, `--max-per-identity`, `--max-samples` clip the iterator without editing YAML.
- `--batch-size` processes multiple images per worker call (helps GPU amortization).
- `--workers` spins up parallel pipelines; each worker loads its own detector/embedder, so size this to available VRAM.

Each experiment folder should reference the configuration it relies upon to keep results reproducible.

### Data configuration

`data.dataset_type` selects the loader (`image_folder`, `celeba`, ...). Use `data.options` to pass loader-specific arguments, for example:

```yaml
data:
	dataset_path: data/celeba
	dataset_type: celeba
	options:
		image_dir: img_align_celeba
		identity_file: identity_CelebA.txt
		attr_file: list_attr_celeba.txt
		split: train
```

All relative paths are resolved from `dataset_path`, keeping configs portable.

### Quantization modes

`quantization.type` selects the discretization strategy:

- `uniform` (default): scalar binning with step `delta`.
- `simhash`: cosine-aware random-hyperplane hashing. Configure via `quantization.simhash.num_planes` (e.g., 128) and optional `random_seed` / `planes_path` to persist the projection vectors.
- `spherical`: Voronoi/nearest-prototype binning on the unit sphere. Set `quantization.spherical.num_prototypes` plus optional `random_seed` / `prototypes_path` to control the anchor set.

Example snippet:

```yaml
quantization:
	type: simhash
	simhash:
		num_planes: 128
		random_seed: 42

quantization:
	type: spherical
	spherical:
		num_prototypes: 256
		random_seed: 7
```

SimHash outputs binary vectors (after float cast) that pair well with the existing HMAC seed generator while being more robust to small cosine-angle changes.
Spherical mode instead emits the index of the closest prototype, producing a single discrete ID derived from the embedding direction.
