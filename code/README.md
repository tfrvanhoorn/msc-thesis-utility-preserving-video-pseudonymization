
# Code Workspace

This folder contains all training, inference, evaluation, and utility scripts.

For the full, detailed workflow (containers, dataset prep, training, inference, evaluation, action recognition, emotion recognition), see the repository root README:

- [README.md](../README.md)

## Quickstart (Containers)

```bash
apptainer build cuda11.8.sif ./code/cuda11.8.def
apptainer exec --nv cuda11.8.sif python ./code/src/training.py --help
apptainer exec --nv cuda11.8.sif python ./code/src/infer.py --help
apptainer exec --nv cuda11.8.sif python ./code/src/evaluation.py --help
```

```bash
apptainer build emotion_recognition.sif ./code/emotion_recognition.def
apptainer exec --nv emotion_recognition.sif python ./code/src/emotion_recognition/emotion_recognition_script.py --help
```
