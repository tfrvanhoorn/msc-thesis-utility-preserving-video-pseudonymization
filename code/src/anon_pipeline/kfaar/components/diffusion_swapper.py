from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter
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
        ip_adapter_scale: float = 1.2, # INCREASED: Force the identity to overpower the background
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
        self.pipe.load_ip_adapter(
            ip_adapter_id, 
            subfolder="models", 
            weight_name=ip_adapter_weight
        )
        # Apply the stronger scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        self.pipe.safety_checker = None 
        
    def _create_inner_face_mask(self, h: int, w: int) -> Image.Image:
        """
        Creates an elliptical mask covering the inner face.
        Expanded slightly to give the model room to map new features.
        """
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Expanded the ellipse to cover eyebrows and jawline better
        left, top = w * 0.15, h * 0.15
        right, bottom = w * 0.85, h * 0.95
        draw.ellipse([left, top, right, bottom], fill=255)
        
        # Increased blur radius for a much smoother gradient between skin tones
        return mask.filter(ImageFilter.GaussianBlur(radius=15))

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.clamp(0.0, 1.0)
        return TF.to_pil_image(tensor)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image).to(self.device)

    def swap(self, source_aligned: torch.Tensor, target_aligned: torch.Tensor) -> torch.Tensor | None:
        try:
            # Range Fix
            if source_aligned.min() < 0.0: source_aligned = (source_aligned + 1.0) / 2.0
            if target_aligned.min() < 0.0: target_aligned = (target_aligned + 1.0) / 2.0

            c, h, w = target_aligned.shape

            source_pil = self._tensor_to_pil(source_aligned)
            target_pil = self._tensor_to_pil(target_aligned)
            
            # Resize target to 512x512
            target_512 = target_pil.resize((512, 512), Image.Resampling.LANCZOS)
            mask_512 = self._create_inner_face_mask(512, 512)

            generator = torch.Generator(device=self.device).manual_seed(42)

            with torch.autocast("cuda"):
                result_512 = self.pipe(
                    # ADDED PROMPT: Give the U-Net context so it knows to draw a face!
                    prompt="A photorealistic high-quality portrait of a person's face, natural skin, seamless blend, highly detailed",
                    negative_prompt="bad anatomy, distorted face, blurry, malformed, poorly drawn, artificial, mismatched skin tone",
                    image=target_512,
                    mask_image=mask_512,
                    ip_adapter_image=source_pil,
                    num_inference_steps=self.inference_steps,
                    guidance_scale=6.5, # Balance between prompt and IP-adapter
                    strength=0.99,      # FORCED: Tells the pipeline to completely replace the white mask area
                    generator=generator,
                    output_type="pil"
                ).images[0]

            result_original_size = result_512.resize((w, h), Image.Resampling.LANCZOS)
            return self._pil_to_tensor(result_original_size)

        except Exception as e:
            import logging
            logging.error(f"Diffusion FaceSwap failed: {e}")
            return None