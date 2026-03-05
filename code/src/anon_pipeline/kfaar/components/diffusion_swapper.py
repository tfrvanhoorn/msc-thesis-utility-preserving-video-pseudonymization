from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting, DDIMScheduler

class DiffusionFaceSwapper:
    """
    A Diffusion-based face swapper replacing SimSwap.
    Uses Stable Diffusion Inpainting + IP-Adapter to seamlessly blend 
    the StyleGAN pseudonym face into the original target context.
    """

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-inpainting",
        ip_adapter_id: str = "h94/IP-Adapter",
        ip_adapter_weight: str = "ip-adapter-plus-face_sd15.bin",
        device: torch.device | str = "cuda:0",
        inference_steps: int = 25,
        ip_adapter_scale: float = 0.7,
        **kwargs
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Diffusion requires CUDA.")

        self.device = torch.device(device)
        self.inference_steps = inference_steps

        # 1. Load the Inpainting Pipeline
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        # Use DDIM for stable, deterministic generation
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # 2. Load the IP-Adapter (Face Plus version)
        # This tells the model to use the source image as the primary identity condition
        self.pipe.load_ip_adapter(
            ip_adapter_id, 
            subfolder="models", 
            weight_name=ip_adapter_weight
        )
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Disable safety checker to prevent false positives during batch processing
        self.pipe.safety_checker = None 
        
    def _create_inner_face_mask(self, h: int, w: int) -> Image.Image:
        """
        Creates an elliptical mask covering the inner face (eyes, nose, mouth).
        This protects the original hair, ears, and background from being altered.
        """
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Define an ellipse that covers the center ~60% of the image
        left, top = w * 0.2, h * 0.2
        right, bottom = w * 0.8, h * 0.9
        draw.ellipse([left, top, right, bottom], fill=255)
        
        # Blur the mask slightly for a seamless blend
        # Note: diffusers inpainting pipeline does some soft blending automatically,
        # but a pre-blurred mask helps.
        from PIL import ImageFilter
        return mask.filter(ImageFilter.GaussianBlur(radius=10))

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts a [C, H, W] tensor in [0, 1] to a PIL Image."""
        tensor = tensor.clamp(0.0, 1.0)
        return TF.to_pil_image(tensor)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image back to a [C, H, W] tensor in [0, 1]."""
        return TF.to_tensor(image).to(self.device)

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> torch.Tensor | None:
        """
        Inpaints the source identity (StyleGAN face) into the target context.
        
        Args:
            source_aligned: The pseudonym face [C, H, W] in [0, 1]
            target_aligned: The original cropped frame [C, H, W] in [0, 1]
        """
        try:
            # Range Fix
            if source_aligned.min() < 0.0: source_aligned = (source_aligned + 1.0) / 2.0
            if target_aligned.min() < 0.0: target_aligned = (target_aligned + 1.0) / 2.0

            c, h, w = target_aligned.shape

            # 1. Prepare Inputs for Diffusers
            source_pil = self._tensor_to_pil(source_aligned)
            target_pil = self._tensor_to_pil(target_aligned)
            
            # Resize target to 512x512 (Diffusion's native resolution) to prevent artifacts
            target_512 = target_pil.resize((512, 512), Image.Resampling.LANCZOS)
            mask_512 = self._create_inner_face_mask(512, 512)

            # 2. Run Diffusion Inpainting
            # - prompt is empty because IP-Adapter is driving the generation
            # - image is the original context
            # - ip_adapter_image is the StyleGAN identity
            generator = torch.Generator(device=self.device).manual_seed(42) # For temporal stability

            with torch.autocast("cuda"):
                result_512 = self.pipe(
                    prompt="",
                    negative_prompt="bad anatomy, distorted face, blurry, malformed",
                    image=target_512,
                    mask_image=mask_512,
                    ip_adapter_image=source_pil,
                    num_inference_steps=self.inference_steps,
                    generator=generator,
                    output_type="pil"
                ).images[0]

            # 3. Restore to original dimensions and tensor format
            result_original_size = result_512.resize((w, h), Image.Resampling.LANCZOS)
            swapped_tensor = self._pil_to_tensor(result_original_size)

            return swapped_tensor

        except Exception as e:
            import logging
            logging.error(f"Diffusion FaceSwap failed: {e}")
            return None