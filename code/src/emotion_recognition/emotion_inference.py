"""
EMO-AffectNetModel inference wrapper.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionInferenceEngine:
    EMOTION_CLASSES = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]

    def __init__(
        self,
        backbone_checkpoint: str,
        lstm_checkpoint: str,
        emo_affectnet_root: Optional[str] = None,
        confidence_threshold: float = 0.7,
        device: str = "cuda",
    ):
        self.backbone_checkpoint = Path(backbone_checkpoint)
        self.lstm_checkpoint = Path(lstm_checkpoint)
        self.confidence_threshold = confidence_threshold
        self.device = device

        if not self.backbone_checkpoint.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {self.backbone_checkpoint}")
        if not self.lstm_checkpoint.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {self.lstm_checkpoint}")

        torch_exts = {".pt", ".pth", ".pth.tar", ".jit"}
        self.use_torch = (
            self.backbone_checkpoint.suffix.lower() in torch_exts
            or self.lstm_checkpoint.suffix.lower() in torch_exts
        )
        if self.use_torch:
            if self.backbone_checkpoint.suffix.lower() not in torch_exts or self.lstm_checkpoint.suffix.lower() not in torch_exts:
                raise ValueError("Torch backend requires both backbone and LSTM checkpoints to be .pt/.pth")

        if emo_affectnet_root is None:
            possible_paths = [
                Path(__file__).parent.parent.parent / "external_libraries" / "EMO-AffectNetModel",
                Path.cwd() / "external_libraries" / "EMO-AffectNetModel",
            ]
            emo_affectnet_root = None
            for candidate in possible_paths:
                if (candidate / "functions").exists():
                    emo_affectnet_root = candidate
                    break
            if emo_affectnet_root is None:
                raise RuntimeError("Could not locate EMO-AffectNetModel repo. Please provide emo_affectnet_root.")

        self.emo_affectnet_root = Path(emo_affectnet_root)
        if str(self.emo_affectnet_root) not in sys.path:
            sys.path.insert(0, str(self.emo_affectnet_root))

        from functions.get_models import load_weights_EE, load_weights_LSTM
        from functions.get_face_areas import VideoCamera
        from functions import sequences as seq_module

        self.load_weights_EE = load_weights_EE
        self.load_weights_LSTM = load_weights_LSTM
        self.VideoCamera = VideoCamera
        self.seq_module = seq_module
        self._load_models()

    def _load_models(self):
        logger.info("Loading backbone model...")
        if self.use_torch:
            import torch

            self.torch = torch
            self.backbone_model = torch.jit.load(str(self.backbone_checkpoint), map_location=self.device)
            self.backbone_model.eval()
            logger.info("Loading LSTM model...")
            self.lstm_model = torch.jit.load(str(self.lstm_checkpoint), map_location=self.device)
            self.lstm_model.eval()
        else:
            self.backbone_model = self.load_weights_EE(str(self.backbone_checkpoint))
            logger.info("Loading LSTM model...")
            self.lstm_model = self.load_weights_LSTM(str(self.lstm_checkpoint))

    def process_video(self, video_path: str) -> Dict:
        import pandas as pd

        video_path = Path(video_path)
        try:
            vc = self.VideoCamera(str(video_path), conf=self.confidence_threshold)
            face_dict, total_frame = vc.get_frame()
            if not face_dict:
                return {"success": False, "error": "No faces detected in video", "frame_count": 0, "detected_faces": 0}

            all_paths = list(face_dict.keys())
            all_faces = list(face_dict.values())
            face_images = np.stack(all_faces, axis=0)
            if self.use_torch:
                torch = self.torch
                face_tensor = torch.from_numpy(face_images).to(self.device, dtype=torch.float32)
                if face_tensor.ndim == 4:
                    face_tensor = face_tensor.permute(0, 3, 1, 2).contiguous()
                with torch.no_grad():
                    features = self.backbone_model(face_tensor)
                features = features.detach().cpu()
            else:
                features = self.backbone_model.predict(face_images, verbose=0)

            seq_paths, seq_features = self.seq_module.sequences(all_paths, features, win=10, step=5)
            if len(seq_features) == 0:
                return {"success": False, "error": "Could not create sequences", "frame_count": len(all_paths), "detected_faces": len(all_faces)}

            if self.use_torch:
                torch = self.torch
                seq_features_tensor = torch.tensor(seq_features, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    lstm_output = self.lstm_model(seq_features_tensor)
                    if isinstance(lstm_output, (tuple, list)):
                        lstm_output = lstm_output[0]
                    lstm_output = torch.softmax(lstm_output, dim=1)
                lstm_predictions = lstm_output.detach().cpu().numpy()
            else:
                seq_features_array = np.array(seq_features, dtype=np.float32)
                lstm_predictions = self.lstm_model.predict(seq_features_array, verbose=0)

            label_model = self.EMOTION_CLASSES
            all_pred = []
            all_path = []
            for index, current_path in enumerate(seq_paths):
                current_frames = [str(i).zfill(6) for i in range(int(current_path[0]), int(current_path[-1]) + 1)]
                current_predictions = [lstm_predictions[index]] * len(current_frames)
                all_pred.extend(current_predictions)
                all_path.extend(current_frames)

            if all_path:
                last_frame = int(all_path[-1])
                missing_frames = [str(i).zfill(6) for i in range(last_frame + 1, total_frame + 1)]
                missing_predictions = [all_pred[-1]] * len(missing_frames)
            else:
                missing_frames = []
                missing_predictions = []

            df = pd.DataFrame(data=all_pred + missing_predictions, columns=label_model)
            df["frame"] = all_path + missing_frames
            df = df[["frame"] + label_model]
            df_grouped = self.seq_module.df_group(df, label_model)

            frame_predictions = []
            frame_probabilities = []
            for _, row in df_grouped.iterrows():
                frame_probs = row[label_model].values.astype(np.float32)
                frame_probs = frame_probs / (frame_probs.sum() + 1e-8)
                frame_predictions.append(int(np.argmax(frame_probs)))
                frame_probabilities.append(frame_probs.tolist())

            avg_probabilities = np.mean(frame_probabilities, axis=0) if frame_probabilities else np.ones(7) / 7
            return {
                "success": True,
                "error": None,
                "frame_count": len(all_paths),
                "detected_faces": len(all_faces),
                "total_frames": total_frame,
                "frame_predictions": frame_predictions,
                "sequence_predictions": lstm_predictions.tolist(),
                "emotion_probabilities": frame_probabilities,
                "average_probabilities": avg_probabilities.tolist(),
            }
        except Exception as exc:
            logger.error(f"Error processing video {video_path}: {exc}", exc_info=True)
            return {"success": False, "error": str(exc), "frame_count": 0, "detected_faces": 0}

    def get_predicted_emotion(self, probabilities: List[float]) -> str:
        if len(probabilities) != 7:
            raise ValueError(f"Expected 7 probabilities, got {len(probabilities)}")
        return self.EMOTION_CLASSES[int(np.argmax(probabilities))]
