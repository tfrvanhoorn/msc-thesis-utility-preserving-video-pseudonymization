# MSc Thesis: Utility-Preserving Video Pseudonymization

This repository contains the full engineering workflow for utility-preserving face pseudonymization in video:

1. Prepare datasets into a canonical naming contract.
2. Train the KFAAR projector model.
3. Run inference to generate anonymized videos (optionally with postprocessing for visuals).
4. Evaluate anonymization, synchronism, diversity, differentiation, detection, and perceptual utility.
5. Run downstream utility checks: action recognition and emotion recognition.

## Repository Structure

- `code/`: model, data pipeline, training, inference, evaluation, and utilities.
- `thesis/`: thesis-writing assets.
- `notes/`: planning and meeting notes.

## Containers (Apptainer)

All experiments in this project were run inside Apptainer containers built from the definition files in `code/`.
Use CUDA 11.8 for training, inference, evaluation, and action recognition. Use the dedicated emotion container for emotion recognition. A CPU-only FaceQnet container is also provided.

### Build Containers

```bash
apptainer build cuda11.8.sif ./code/cuda11.8.def
apptainer build emotion_recognition.sif ./code/emotion_recognition.def
apptainer build faceqnet_evaluation.sif ./code/faceqnet_evaluation.def
```

### Run Containers (GPU)

```bash
apptainer exec --nv cuda11.8.sif python --version
apptainer exec --nv emotion_recognition.sif python --version
```

### Run Containers (CPU)

```bash
apptainer exec faceqnet_evaluation.sif python --version
```

### Which Container To Use

- Training, inference, evaluation, action recognition: `cuda11.8.sif`
- Emotion recognition: `emotion_recognition.sif`
- Optional FaceQnet-only evaluation: `faceqnet_evaluation.sif`

## Dataset Preparation (Canonical Naming Contract)

All inference and evaluation steps rely on one deterministic filename format:

`{id}_sample{count}_{original_filename}.mp4`

Example: `id01234_sample3_00027.mp4`

### Contract Rules

1. `id` cannot contain underscores.
2. `count` is an integer starting at 1 and incremented per identity.
3. `original_filename` is free-form and can contain underscores.
4. Prepared videos are MP4. Prepared images keep their original image extension.

### Entry Point

`code/src/dataset_utils.py`

### Supported Preparation Types

- `voxceleb`: uses the VoxCeleb `dev/mp4/<identity>/<youtube_id>/*.mp4` layout.
- `video_folder`: recursively scans a folder and assigns a single shared identity.
- `celeba`: supports identity subfolders or the flat CelebA layout with identity mapping.

### Required Arguments

1. `--type voxceleb|video_folder|celeba`
2. `--data_path <source_dataset_root>`
3. `--output_dir <prepared_output_dir>`

### Optional Filters And Sampling

- `--max_identities`
- `--max_videos_per_youtube_id` (voxceleb)
- `--max_videos_per_id`
- `--max_frames_per_video`
- `--fps`
- `--video_folder_identity` (video_folder)
- `--video_folder_preserve_subfolders` (video_folder)
- `--celeba_identity_file` and `--celeba_images_subdir` (celeba flat layout)

### VoxCeleb Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/dataset_utils.py \
	--type voxceleb \
	--data_path /data/voxceleb2 \
	--output_dir /data/prepared_inputs \
	--max_identities 100 \
	--max_videos_per_youtube_id 3 \
	--max_videos_per_id 10 \
	--max_frames_per_video 64 \
	--fps 10
```

### Video Folder Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/dataset_utils.py \
	--type video_folder \
	--data_path /data/my_videos \
	--output_dir /data/prepared_inputs \
	--video_folder_identity video \
	--video_folder_preserve_subfolders \
	--max_frames_per_video 64 \
	--fps 10
```

### CelebA Example (Identity Subfolders)

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/dataset_utils.py \
	--type celeba \
	--data_path /data/celeba_by_identity \
	--output_dir /data/prepared_celeba_train \
	--max_identities 2000 \
	--max_videos_per_id 50
```

### CelebA Example (Flat Layout + Identity Map)

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/dataset_utils.py \
	--type celeba \
	--data_path /data/celeba \
	--output_dir /data/prepared_celeba_train \
	--celeba_identity_file identity_CelebA.txt \
	--celeba_images_subdir img_align_celeba \
	--max_identities 2000 \
	--max_videos_per_id 50
```

## Training (KFAAR Projector)

Entry point: `code/src/training.py`

### Minimal Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/training.py \
	--input_dir /data/prepared_celeba_train \
	--output_dir ./code/train_results \
	--epochs 10 \
	--max_samples_per_identity 50
```

### Common Options

- Projector: `--key_dim`, `--enable_projector_l2_reg`, `--enable_projector_key_upscaler`, `--use_stylegan_mapper`
- Loss weights: `--lambda_ano`, `--lambda_syn`, `--lambda_div`, `--lambda_dif`, `--lambda_temp`, `--lambda_w_reg`
- Resume: `--resume_ckpt` and `--start_epoch`
- Generated faces: `--save_generated_faces`, `--save_generated_mode`, `--save_generated_dir`

### Notes

1. Training uses prepared image inputs created by `dataset_utils.py`.
2. Identity tokens cannot contain underscores.
3. Pseudonymization keys are sampled as binary 0/1 vectors of length `key_dim`.

## Inference (Generate Anonymized Videos)

Entry point: `code/src/infer.py`

### Required Inputs

- `--checkpoint`
- `--data_path` (prepared input directory)
- `--dataset_type video_folder`
- `--output_dir`

### Output Layout

1. If `--num_keys 1`, outputs are written directly into `output_dir`.
2. If `--num_keys >= 2`, outputs are written under `output_dir/key1`, `output_dir/key2`, ...
3. Each key vector is sampled as a binary 0/1 vector of length `key_dim`.

### Rendering Method (Postprocessing)

Inference supports different face rendering methods for visualization. Use the following flags to control the rendering path:

- `--face_postprocessor none` (default): use StyleGAN output directly.
- `--face_postprocessor crop_bbox`: crop synthesized face before paste-back.
- `--face_postprocessor faceadapter_swap`: use Face-Adapter swapper.
- `--face_postprocessor faceadapter_reenactment`: use Face-Adapter reenactment.

By default, embeddings are computed on the StyleGAN output even if a postprocessor is enabled. To use swapped outputs for embeddings, set `--swap_for_loss` (legacy behavior). For visuals-only swapping, keep the default `--swap_for_visuals_only`.

### Single-Key Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/infer.py \
	--checkpoint ./code/train_results/checkpoint_epoch9.pt \
	--data_path /data/prepared_inputs \
	--dataset_type video_folder \
	--output_dir /data/inferred_single_key \
	--num_keys 1 \
	--max_frames_per_video 64 \
	--target_sample_fps 10
```

### Multi-Key Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/infer.py \
	--checkpoint ./code/train_results/checkpoint_epoch9.pt \
	--data_path /data/prepared_inputs \
	--dataset_type video_folder \
	--output_dir /data/inferred_multi_key \
	--num_keys 2 \
	--max_frames_per_video 64 \
	--target_sample_fps 10
```

### Output Artifacts

- `manifest.json`: per-input output mapping and metadata.
- `<checkpoint>_<dataset>_infer.json`: run report for auditing and reproducibility.

## Evaluation (Anonymization + Utility Metrics)

Entry point: `code/src/evaluation.py`

### Core Required Arguments

- `--input_dir <prepared_input_dir>`
- `--inferred_dir <inferred_output_dir>`
- `--metrics <comma_separated_list>`

### Key Folder Flag

Use `--inferred_nested_keys` when inferred outputs are stored under `key1`, `key2`, ... folders.

### Diversity Constraint

`diversity` can only be computed when:

1. `--inferred_nested_keys` is enabled, and
2. at least two key folders are present.

### Regex Grouping

Evaluation groups and pairs input/inferred videos by regex parsing over filenames. Default pattern:

`^(?P<identity>[^_]+)_sample(?P<sample>\d+)_(?P<original>.+)$`

You can override with `--filename_regex`, but your regex must define named groups: `identity`, `sample`, and `original`.

### Single-Key Evaluation Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/evaluation.py \
	--input_dir /data/prepared_inputs \
	--inferred_dir /data/inferred_single_key \
	--metrics detection,anonymization,synchronism,differentiation,landmark_distance,lpips,ssim \
	--detection_key 1 \
	--output_dir ./code/eval_results
```

### Multi-Key Evaluation Example (with Diversity)

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/evaluation.py \
	--input_dir /data/prepared_inputs \
	--inferred_dir /data/inferred_multi_key \
	--inferred_nested_keys \
	--metrics detection,anonymization,synchronism,diversity,differentiation,landmark_distance,lpips,ssim \
	--detection_key 1 \
	--output_dir ./code/eval_results
```

### Output Artifacts

- JSON summary metrics and per-sample counts written under `--output_dir`.

## Action Recognition (UCF101)

Entry point: `code/src/action_recognition/evaluate_ucf101.py`

### Expected Input Layout

```
<input_dir>/
	<label_name_1>/
		*.mp4
	<label_name_2>/
		*.mp4
```

Label folder names are matched against the model label set after normalization (case/spacing/underscore differences are handled). Unknown labels are reported and skipped.

### Example

```bash
apptainer exec --nv cuda11.8.sif python ./code/src/action_recognition/evaluate_ucf101.py \
	--input_dir /data/ucf101_eval \
	--output_dir ./code/action_eval_results \
	--model_id nateraw/videomae-base-finetuned-ucf101 \
	--video_extensions "*.mp4,*.avi" \
	--recursive
```

### Output Artifacts

- JSON report containing per-video predictions, per-class accuracy, and overall metrics.

## Emotion Recognition (RAVDESS)

Entry point: `code/src/emotion_recognition/emotion_recognition_script.py`

### Required Inputs

- `--input-dir`: RAVDESS videos (keyed or unkeyed).
- `--backbone-checkpoint`: ResNet50 backbone checkpoint.
- `--lstm-checkpoint`: temporal model checkpoint.
- `--output-json`: output report path.

### Keyed vs Unkeyed Inputs

- Unkeyed: `--input-dir` points to a folder of RAVDESS videos.
- Keyed: `--input-dir` points to a folder containing `key1`, `key2`, ... subfolders. Use `--inferred-keyed-dir` and `--num-keys`.

### Example (Unkeyed)

```bash
apptainer exec --nv emotion_recognition.sif python ./code/src/emotion_recognition/emotion_recognition_script.py \
	--input-dir /data/ravdess_videos \
	--backbone-checkpoint /models/emo_backbone.pt \
	--lstm-checkpoint /models/emo_lstm.pt \
	--output-json ./code/emotion_eval_results/emotion_report.json
```

### Example (Keyed)

```bash
apptainer exec --nv emotion_recognition.sif python ./code/src/emotion_recognition/emotion_recognition_script.py \
	--input-dir /data/ravdess_keyed \
	--backbone-checkpoint /models/emo_backbone.pt \
	--lstm-checkpoint /models/emo_lstm.pt \
	--output-json ./code/emotion_eval_results/emotion_report.json \
	--inferred-keyed-dir \
	--num-keys 2
```

### Output Artifacts

- JSON report with per-video predictions, key-wise summaries, confidence shifts, and agreement metrics.

## TODO: Pretrained Models

- TODO: Add download links for StyleGAN2 checkpoints, projector checkpoints, and emotion recognition backbone/LSTM weights.

## End-To-End Quick Flow

1. Prepare datasets with `dataset_utils.py`.
2. Train the projector with `training.py`.
3. Run inference with `infer.py` to generate anonymized videos.
4. Evaluate with `evaluation.py`.
5. Run action and emotion recognition as downstream utility checks.

## Troubleshooting

1. Error: filename does not match prepared convention.
	 Fix: regenerate inputs with `dataset_utils.py` or rename to match `{id}_sample{count}_{original_filename}.mp4`.
2. Error: diversity requires nested keys.
	 Fix: rerun inference with `--num_keys >= 2` and evaluate with `--inferred_nested_keys`.
3. Error: missing inferred output pairs.
	 Fix: ensure inference completed for all prepared input files and key folders.
