# MSc Thesis: Utility-Preserving Video Anonymization

This repository contains the full engineering workflow for utility-preserving face anonymization in video:

1. Add/organize raw datasets.
2. Train a projector model.
3. Prepare inference/evaluation input videos with a canonical naming contract.
4. Run inference to generate anonymized videos.
5. Evaluate anonymization, synchronism, diversity, differentiation, detection, and geometric utility.

## Repository Structure

- `code/`: model, data pipeline, training, inference, evaluation, and utilities.
- `thesis/`: thesis-writing assets.
- `notes/`: planning and meeting notes.

## Environment Setup

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\code\requirements.txt
```

If you use external swappers (`SimSwap`, `Face-Adapter`), install their extra requirements as needed.

## Dataset Onboarding

### Supported Training Dataset Types

Training (`code/src/training.py`) currently supports prepared CelebA image inputs only.

Prepare your dataset first with `code/src/dataset_utils.py` and then point training to that prepared folder.

### Generic Rule To Add A New Dataset

Map your raw dataset into either:

1. A training loader format (`image_folder`, `video_folder`, or a new loader), and
2. The prepared filename contract for inference/evaluation.

For inference/evaluation specifically, the system only needs `id`, `sample`, and `original_filename` encoded in the filename.

## Prepare VoxCeleb For Inference/Evaluation

Use the new utility script:

`code/src/dataset_utils.py`

### Required Arguments

1. `--type voxceleb`, `--type video_folder`, or `--type celeba`
2. `--data_path <source_dataset_root>`
3. `--output_dir <prepared_input_dir>`

### Optional Filtering/Sampling Arguments

1. `--max_identities`
2. `--max_videos_per_youtube_id`
3. `--max_videos_per_id`
4. `--max_frames_per_video`
5. `--fps`
6. `--video_folder_identity` (only used when `--type video_folder`)
7. `--celeba_identity_file` and `--celeba_images_subdir` (only used when `--type celeba` with flat + identity-map layout)

### Example

```powershell
python .\code\src\dataset_utils.py \
	--type voxceleb \
	--data_path D:\datasets\voxceleb2 \
	--output_dir D:\datasets\prepared_inputs \
	--max_identities 100 \
	--max_videos_per_youtube_id 3 \
	--max_videos_per_id 10 \
	--max_frames_per_video 64 \
	--fps 10
```

### Video Folder Example

```powershell
python .\code\src\dataset_utils.py \
	--type video_folder \
	--data_path D:\datasets\my_videos \
	--output_dir D:\datasets\prepared_inputs \
	--video_folder_identity video \
	--max_frames_per_video 64 \
	--fps 10
```

When `--type video_folder` is used, videos are discovered recursively and all videos are assigned the same shared identity (default: `video`).

Expected source VoxCeleb layout:

```text
<data_path>/dev/mp4/<identity>/<youtube_id>/*.mp4
```

Prepared output layout:

```text
<output_dir>/
	id00012_sample1_00001.mp4
	id00012_sample2_00005.mp4
	id00013_sample1_00003.mp4
	dataset_preparation_report.json
```

## Prepare CelebA For Training

Use the same preparation utility:

`code/src/dataset_utils.py`

Prepared training inputs use the same naming contract as inference/evaluation, but keep image extensions (`.jpg`, `.jpeg`, `.png`).

### Supported CelebA Source Layouts

1. Identity subfolders:

```text
<data_path>/
	<identity>/
		*.jpg|*.jpeg|*.png
```

2. Flat image folder with identity map file (current CelebA default style):

```text
<data_path>/
	identity_CelebA.txt
	img_align_celeba/
		*.jpg
```

### Identity Subfolders Example

```powershell
python .\code\src\dataset_utils.py \
	--type celeba \
	--data_path D:\datasets\celeba_by_identity \
	--output_dir D:\datasets\prepared_celeba_train \
	--max_identities 2000 \
	--max_videos_per_id 50
```

### Flat + Identity Map Example

```powershell
python .\code\src\dataset_utils.py \
	--type celeba \
	--data_path D:\datasets\celeba \
	--output_dir D:\datasets\prepared_celeba_train \
	--celeba_identity_file identity_CelebA.txt \
	--celeba_images_subdir img_align_celeba \
	--max_identities 2000 \
	--max_videos_per_id 50
```

## Train The Pipeline

Entry point:

`code/src/training.py`

### Minimal Example

```powershell
python .\code\src\training.py \
	--input_dir D:\datasets\prepared_celeba_train \
	--output_dir .\code\train_results \
	--epochs 10 \
	--max_samples_per_identity 50
```

### Notes

1. Training requires prepared image filenames from `dataset_utils.py`.
2. Identity tokens in prepared filenames cannot contain underscores.
3. Use `--resume_ckpt` to continue from a saved checkpoint.
4. Projector architecture is MLP-only.
5. Pseudonymization keys are sampled as binary 0/1 vectors of length `key_dim`.

## Why The Prepared Naming Contract Exists

Inference and evaluation now rely on one deterministic filename format:

`{id}_sample{count}_{original_filename}.mp4`

Example:

`id01234_sample3_00027.mp4`

This naming contract makes pairing explicit without depending on traversal order or folder heuristics.

### Contract Rules

1. `id` cannot contain underscores.
2. `count` is an integer starting at 1 and incremented per identity.
3. `original_filename` is free-form and can contain underscores.
4. Files are MP4 for the prepared dataset workflow.

## Run Inference On Prepared Inputs

Entry point:

`code/src/infer.py`

### Required Inputs

1. `--checkpoint`
2. `--data_path` pointing to prepared input folder
3. `--dataset_type video_folder`
4. `--output_dir`

### Important Output Rule

1. If `--num_keys 1`: outputs are written directly into `output_dir` with the same prepared filename.
2. If `--num_keys >= 2`: outputs are written into nested key folders `key1`, `key2`, ... with the same prepared filename in each key folder.
3. Each key vector is sampled as a binary 0/1 vector of length `key_dim`.

### Single-Key Example

```powershell
python .\code\src\infer.py \
	--checkpoint .\code\train_results\checkpoint_epoch9.pt \
	--data_path D:\datasets\prepared_inputs \
	--dataset_type video_folder \
	--output_dir D:\datasets\inferred_single_key \
	--num_keys 1 \
	--max_frames_per_video 64 \
	--target_sample_fps 10
```

Result:

```text
<output_dir>/
	id00012_sample1_00001.mp4
	id00012_sample2_00005.mp4
	manifest.json
	<checkpoint>_video_folder_infer.json
```

### Multi-Key Example

```powershell
python .\code\src\infer.py \
	--checkpoint .\code\train_results\checkpoint_epoch9.pt \
	--data_path D:\datasets\prepared_inputs \
	--dataset_type video_folder \
	--output_dir D:\datasets\inferred_multi_key \
	--num_keys 2 \
	--max_frames_per_video 64 \
	--target_sample_fps 10
```

Result:

```text
<output_dir>/
	key1/
		id00012_sample1_00001.mp4
		id00012_sample2_00005.mp4
	key2/
		id00012_sample1_00001.mp4
		id00012_sample2_00005.mp4
	manifest.json
	<checkpoint>_video_folder_infer.json
```

## Evaluate Prepared Inputs And Inferred Outputs

Entry point:

`code/src/evaluation.py`

### Core Required Arguments

1. `--input_dir <prepared_input_dir>`
2. `--inferred_dir <inferred_output_dir>`
3. `--metrics ...`

### Key Folder Flag

Use `--inferred_nested_keys` when inferred outputs are stored under `key1`, `key2`, ... folders.

### Diversity Constraint

`diversity` can only be computed when:

1. `--inferred_nested_keys` is enabled, and
2. at least two key folders are present.

This is because diversity compares outputs of the same sample across different keys.

### Single-Key Evaluation Example

```powershell
python .\code\src\evaluation.py \
	--input_dir D:\datasets\prepared_inputs \
	--inferred_dir D:\datasets\inferred_single_key \
	--metrics detection,anonymization,synchronism,differentiation,geometric \
	--detection_key 1 \
	--output_dir .\code\eval_results
```

### Multi-Key Evaluation Example (with diversity)

```powershell
python .\code\src\evaluation.py \
	--input_dir D:\datasets\prepared_inputs \
	--inferred_dir D:\datasets\inferred_multi_key \
	--inferred_nested_keys \
	--metrics detection,anonymization,synchronism,diversity,differentiation,geometric \
	--detection_key 1 \
	--output_dir .\code\eval_results
```

### Regex Grouping

Evaluation groups and pairs input/inferred videos by regex parsing over filenames. The default pattern is:

`^(?P<identity>[^_]+)_sample(?P<sample>\d+)_(?P<original>.+)$`

You can override with `--filename_regex`, but your regex must define named groups:

1. `identity`
2. `sample`
3. `original`

## End-To-End Quick Flow

1. Train model on raw dataset with `training.py`.
2. Prepare inference/evaluation inputs with `dataset_utils.py`.
3. Run inference with `infer.py` into an inferred output directory.
4. Run `evaluation.py` with explicit input and inferred directories.

## Troubleshooting

1. Error: filename does not match prepared convention.
Cause: input/inferred filename not in `{id}_sample{count}_{original_filename}.mp4` format.
Fix: regenerate inputs using `dataset_utils.py` or rename files to match.

2. Error: diversity requires nested keys.
Cause: requested `diversity` without `--inferred_nested_keys` or with fewer than two key folders.
Fix: rerun inference with `--num_keys >= 2` and evaluate with `--inferred_nested_keys`.

3. Missing inferred output pairs.
Cause: not all input samples have corresponding outputs per required key.
Fix: ensure inference completed for all prepared input files and key folders.

## Practical Design Rationale

1. Separate `input_dir` and `inferred_dir` decouple original and anonymized assets.
2. Regex-based grouping prevents accidental pairing by file iteration order.
3. Key folders (`key1`, `key2`, ...) make cross-key metrics (diversity) explicit and reproducible.
4. Prepared naming supports general datasets beyond VoxCeleb by standardizing identity/sample semantics.