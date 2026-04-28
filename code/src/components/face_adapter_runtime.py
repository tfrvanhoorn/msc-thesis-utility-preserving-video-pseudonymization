from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _load_module_from_file(module_name: str, module_file: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {module_name} from {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FaceAdapterRuntime:
    """Shared FaceAdapter runtime utilities for FaceSwap and reenactment modes."""

    def __init__(
        self,
        faceadapter_root: str | Path,
        checkpoint_dir: str | Path | None = None,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        cache_dir: str | Path | None = None,
        use_cache: bool = False,
        device: torch.device | str = "cuda:0",
        inference_steps: int = 25,
        guidance_scale: float = 5.0,
        crop_ratio: float = 0.81,
        detector_name: str = "antelopev2",
        detector_size: int = 640,
        seed: int = 0,
        download_if_missing: bool = True,
        detector_score_threshold: float = 0.55,
        detector_min_face_size: int = 20,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FaceAdapter postprocessing requires CUDA.")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("FaceAdapter postprocessing only supports CUDA devices.")
        if self.device.index is None:
            self.device = torch.device("cuda:0")

        self.faceadapter_root = Path(faceadapter_root).resolve()
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else self.faceadapter_root / "checkpoints"
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else self.faceadapter_root / "hub"
        self.use_cache = bool(use_cache)
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.crop_ratio = float(crop_ratio)
        self.seed = int(seed)
        self.test_image_size = 512
        self.weight_dtype = torch.float16
        self.detector_score_threshold = float(detector_score_threshold)
        self.detector_min_face_size = int(detector_min_face_size)
        self.last_failure_reasons: list[str | None] = []

        if str(self.faceadapter_root) not in sys.path:
            sys.path.insert(0, str(self.faceadapter_root))

        try:
            snapshot_download = importlib.import_module("huggingface_hub").snapshot_download

            if download_if_missing and not self.checkpoint_dir.exists():
                snapshot_download(repo_id="FaceAdapter/FaceAdapter", local_dir=str(self.checkpoint_dir))

            set_seed = importlib.import_module("accelerate.utils").set_seed
            diffusers_mod = importlib.import_module("diffusers")
            transformers_mod = importlib.import_module("transformers")
            FaceAnalysis = importlib.import_module("insightface.app").FaceAnalysis

            AutoencoderKL = diffusers_mod.AutoencoderKL
            ControlNetModel = diffusers_mod.ControlNetModel
            EulerDiscreteScheduler = diffusers_mod.EulerDiscreteScheduler
            UNet2DConditionModel = diffusers_mod.UNet2DConditionModel
            CLIPImageProcessor = transformers_mod.CLIPImageProcessor
            CLIPVisionModel = transformers_mod.CLIPVisionModel

            datasets_faceswap = _load_module_from_file(
                "faceadapter_datasets_faceswap",
                self.faceadapter_root / "data" / "datasets_faceswap.py",
            )
            model_seg_unet = importlib.import_module("face_adapter.model_seg_unet")
            bfm = importlib.import_module("third_party.d3dfr.bfm")
            model_insightface_backbone = importlib.import_module("third_party.insightface_backbone_conv")
            model_resnet_d3dfr = importlib.import_module("third_party.model_resnet_d3dfr")

            model_to_token = importlib.import_module("face_adapter.model_to_token")
            ID2Token = model_to_token.ID2Token
            Image2Token = model_to_token.Image2Token

            face_adapter_pipeline_mod = importlib.import_module("face_adapter_pipline")
            StableDiffusionFaceAdapterPipeline = face_adapter_pipeline_mod.StableDiffusionFaceAdapterPipeline
            draw_pts70_batch = face_adapter_pipeline_mod.draw_pts70_batch

            StableDiffusionFaceAdapterPipeline.check_inputs = lambda *args, **kwargs: None

            self._set_seed = set_seed
            self._datasets_faceswap = datasets_faceswap
            self._draw_pts70_batch = draw_pts70_batch

            controlnet = ControlNetModel.from_pretrained(
                str(self.checkpoint_dir / "controlnet"),
                torch_dtype=self.weight_dtype,
            ).to(self.device)

            self.pipe = StableDiffusionFaceAdapterPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=self.weight_dtype,
                cache_dir=str(self.cache_dir) if self.use_cache else None,
                local_files_only=self.use_cache,
                requires_safety_checker=False,
            ).to(self.device)

            pretrained_unet_path = self.checkpoint_dir / "pretrained_unet"
            if pretrained_unet_path.exists():
                self.pipe.unet = UNet2DConditionModel.from_pretrained(
                    str(pretrained_unet_path),
                    torch_dtype=self.weight_dtype,
                ).to(self.device)

            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

            self.pipe.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                cache_dir=str(self.cache_dir) if self.use_cache else None,
                torch_dtype=self.weight_dtype,
                local_files_only=self.use_cache,
            ).to(self.device)

            self.net_d3dfr = model_resnet_d3dfr.getd3dfr_res50(
                str(self.checkpoint_dir / "third_party" / "d3dfr_res50_nofc.pth")
            ).eval().to(self.device)
            self.bfm_facemodel = bfm.BFM(
                focal=1015 * 256 / 224,
                image_size=256,
                bfm_model_path=str(self.checkpoint_dir / "third_party" / "BFM_model_front.mat"),
            ).to(self.device)
            self.net_arcface = model_insightface_backbone.getarcface(
                str(self.checkpoint_dir / "third_party" / "insightface_glint360k.pth")
            ).to(self.device)

            self.clip_image_processor = CLIPImageProcessor()
            self.net_vision_encoder = CLIPVisionModel.from_pretrained(
                str(self.checkpoint_dir / "vision_encoder")
            ).eval().to(self.device)

            self.net_image2token = Image2Token(
                visual_hidden_size=self.net_vision_encoder.vision_model.config.hidden_size,
                text_hidden_size=768,
                max_length=77,
                num_layers=3,
            ).to(self.device)
            self.net_image2token.load_state_dict(
                torch.load(
                    self.checkpoint_dir / "net_image2token.pth",
                    map_location=self.device,
                    weights_only=False,
                )
            )
            self.net_image2token.eval()

            self.net_id2token = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3).to(self.device)
            self.net_id2token.load_state_dict(
                torch.load(
                    self.checkpoint_dir / "net_id2token.pth",
                    map_location=self.device,
                    weights_only=False,
                )
            )
            self.net_id2token.eval()

            self.net_seg_res18 = model_seg_unet.UNet().eval().to(self.device)
            self.net_seg_res18.load_state_dict(
                torch.load(
                    self.checkpoint_dir / "net_seg_res18.pth",
                    map_location=self.device,
                    weights_only=False,
                )
            )

            model_parsing = importlib.import_module("third_party.model_parsing")
            self.net_seg_parsing = model_parsing.get_face_parsing(
                str(self.checkpoint_dir / "third_party" / "79999_iter.pth")
            ).eval().to(self.device)

            self.coco_seg_model = YOLO("yolov8n-seg.pt")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.app = FaceAnalysis(name=detector_name, root=str(self.checkpoint_dir / "third_party"), providers=providers)
            self.app.prepare(ctx_id=self.device.index or 0, det_size=(detector_size, detector_size))

            empty_prompt_path = self.faceadapter_root / "empty_prompt_embedding.pth"
            if not empty_prompt_path.exists():
                empty_prompt_path = self.checkpoint_dir / "empty_prompt_embedding.pth"
            if not empty_prompt_path.exists():
                raise FileNotFoundError("Missing empty prompt embedding file")
            self.empty_prompt_token = torch.load(
                str(empty_prompt_path),
                map_location=self.device,
                weights_only=False,
            ).view(1, 77, 768).to(dtype=self.weight_dtype, device=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize FaceAdapter runtime: {exc}") from exc

        self.pil2tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    @staticmethod
    def _largest_face(face_infos: list) -> dict:
        if not face_infos:
            raise RuntimeError("No face detected")
        return sorted(
            face_infos,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]

    def _filter_face_infos(self, face_infos: list) -> list:
        filtered: list = []
        for face in face_infos:
            bbox = face.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            if min(width, height) < float(self.detector_min_face_size):
                continue
            det_score = float(face.get("det_score", 1.0))
            if det_score < float(self.detector_score_threshold):
                continue
            filtered.append(face)
        return filtered

    def _detect_face_info(self, image_rgb: np.ndarray) -> dict:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face_info = self._filter_face_infos(self.app.get(bgr))

        if not face_info:
            h, w = bgr.shape[:2]
            pad_h, pad_w = int(h * 0.5), int(w * 0.5)
            padded_bgr = cv2.copyMakeBorder(
                bgr,
                pad_h,
                pad_h,
                pad_w,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=(128, 128, 128),
            )

            face_info = self._filter_face_infos(self.app.get(padded_bgr))
            if not face_info:
                raise RuntimeError("No face detected even after padding.")

            largest_face = self._largest_face(face_info)
            largest_face["bbox"][0] -= pad_w
            largest_face["bbox"][1] -= pad_h
            largest_face["bbox"][2] -= pad_w
            largest_face["bbox"][3] -= pad_h
            largest_face["kps"][:, 0] -= pad_w
            largest_face["kps"][:, 1] -= pad_h
            return largest_face

        return self._largest_face(face_info)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().to("cpu").clamp(0.0, 1.0)
        return TF.to_pil_image(tensor)

    def _reset_failure_reasons(self, count: int) -> None:
        self.last_failure_reasons = [None] * count

    def _set_failure_reason(self, idx: int, reason: str) -> None:
        if 0 <= idx < len(self.last_failure_reasons):
            self.last_failure_reasons[idx] = reason

    def _prepare_aligned_face(self, image_pil: Image.Image, *, role: str) -> dict[str, torch.Tensor | np.ndarray | Image.Image]:
        ds = self._datasets_faceswap
        image_np = np.array(image_pil.convert("RGB"))
        try:
            face_info = self._detect_face_info(image_np)
        except Exception as exc:
            raise RuntimeError(f"No face detected in {role} input") from exc
        dets = face_info["bbox"]

        if self.crop_ratio > 0:
            bbox = dets[0:4]
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            bbox_x = 0.5 * (bbox[2] + bbox[0])
            bbox_y = 0.5 * (bbox[3] + bbox[1])
            x1 = bbox_x - bbox_size * self.crop_ratio
            x2 = bbox_x + bbox_size * self.crop_ratio
            y1 = bbox_y - bbox_size * self.crop_ratio
            y2 = bbox_y + bbox_size * self.crop_ratio
            bbox_pts4 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
        else:
            bbox = dets[0:4].reshape((2, 2))
            bbox_pts4 = ds.get_box_lm4p(bbox)

        warp_mat_crop = ds.transformation_from_points(bbox_pts4, ds.mean_box_lm4p_512)
        image_crop512 = cv2.warpAffine(image_np, warp_mat_crop, (self.test_image_size, self.test_image_size), flags=cv2.INTER_LINEAR)
        image_crop512_pil = Image.fromarray(image_crop512)

        try:
            face_info_512 = self._detect_face_info(image_crop512)
        except Exception as exc:
            raise RuntimeError(f"No face detected in {role} crop512") from exc
        pts5 = face_info_512["kps"]
        warp_mat_256 = ds.get_affine_transform(pts5, ds.mean_face_lm5p_256)
        image_crop256 = cv2.warpAffine(image_crop512, warp_mat_256, (256, 256), flags=cv2.INTER_LINEAR)
        image_crop256_pil = Image.fromarray(image_crop256)

        image_256_t = self.pil2tensor(image_crop256_pil).view(1, 3, 256, 256).to(self.device)
        image_512_t = self.pil2tensor(image_crop512_pil).view(1, 3, self.test_image_size, self.test_image_size).to(self.device)
        clip_t = self.clip_image_processor(images=image_crop512_pil, return_tensors="pt").pixel_values.view(-1, 3, 224, 224).to(self.device)

        return {
            "image_256": image_256_t,
            "image_512": image_512_t,
            "clip": clip_t,
            "warp_mat_256": warp_mat_256.reshape((1, 2, 3)),
            "warp_mat_512": warp_mat_crop,
            "pil_512": image_crop512_pil,
        }

    def _build_swap_mask(self, images_tar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_pred = self.net_seg_parsing(images_tar)[0]
        masks_tar = torch.argmax(
            F.interpolate(seg_pred, [self.test_image_size, self.test_image_size], mode="bilinear", align_corners=False),
            dim=1,
            keepdim=True,
        )

        mask_0_6 = (masks_tar > 0) & (masks_tar < 7)
        mask_9_14 = (masks_tar > 9) & (masks_tar < 14)
        face_masks_tar = torch.logical_or(mask_0_6, mask_9_14).float() - (masks_tar == 6).float()

        mask_0_14 = (masks_tar > 0) & (masks_tar < 14)
        face_masks_tar_withear = mask_0_14.float() - (masks_tar == 9).float() - (masks_tar == 6).float()

        occ_mask = ((masks_tar == 0) | (masks_tar == 6) | (masks_tar == 9) | (masks_tar == 15) |
            (masks_tar == 18) | (masks_tar == 16) | (masks_tar == 17)).float()
        face_masks_tar = torch.max(face_masks_tar_withear, F.max_pool2d(face_masks_tar, kernel_size=65, stride=1, padding=32))
        face_masks_tar = face_masks_tar * (1 - occ_mask)
        face_masks_tar = F.max_pool2d(face_masks_tar, kernel_size=5, stride=1, padding=2)

        face_masks_tar_pad = F.pad(face_masks_tar, (16, 16, 16, 16), "constant", 0)
        blend_mask = F.max_pool2d(face_masks_tar_pad, kernel_size=17, stride=1, padding=8)
        blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
        blend_mask = blend_mask[:, :, 16:528, 16:528]

        return face_masks_tar, blend_mask

    def swap_batch(self, source_aligned_batch: list[torch.Tensor], target_aligned_batch: list[torch.Tensor]) -> list[torch.Tensor | None]:
        raise NotImplementedError

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> torch.Tensor | None:
        return self.swap_batch([source_aligned], [target_aligned])[0]
