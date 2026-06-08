"""Google SigLIP 2 vision encoder for lip/video frame features.

Replaces the lipreading CNN-ResNet (256-d per frame) with SigLIP 2 image
embeddings. Each video frame is encoded independently via
``get_image_features``, producing a sequence ``[T, D]`` compatible with the
MISP-QEKS dataloader (expects 2-D ``[num_frames, feat_dim]`` arrays).

Requires ``transformers>=4.49`` with SigLIP 2 support:
  pip install "transformers>=4.49.0"

Reference: https://huggingface.co/google/siglip2-base-patch16-224
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
# Native hidden size for google/siglip2-base-patch16-224 (ViT-B).
SIGLIP2_BASE_FEAT_DIM = 768
# Downstream XEQ-Matcher Vide_Proj expects 256-d inputs by default.
MATCHER_VIDEO_FEAT_DIM = 256


def _frames_to_pil_list(frames: torch.Tensor) -> list[Image.Image]:
    """Convert ``[T, H, W, C]`` uint8/float tensor to PIL RGB images."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames [T, H, W, 3], got {tuple(frames.shape)}")

    frames_u8 = frames.detach().cpu()
    if frames_u8.dtype != torch.uint8:
        if frames_u8.max() <= 1.0:
            frames_u8 = (frames_u8.clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            frames_u8 = frames_u8.to(torch.uint8)

    images = []
    for t in range(frames_u8.shape[0]):
        images.append(Image.fromarray(frames_u8[t].numpy(), mode="RGB"))
    return images


def _load_siglip2(model_id: str, device: torch.device):
    """Load SigLIP 2 vision model + processor with version fallbacks."""
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for SigLIP 2. "
            'Install with: pip install "transformers>=4.49.0"'
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    if not hasattr(model, "get_image_features"):
        raise RuntimeError(
            f"Loaded model {model_id} does not expose get_image_features(). "
            "Upgrade transformers to a SigLIP 2 capable release (>=4.49)."
        )
    model.eval()
    model.to(device)
    return model, processor


class Siglip2VideoEncoder:
    """Frame-wise SigLIP 2 encoder with optional projection to matcher dim."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | torch.device | None = None,
        output_dim: int | None = MATCHER_VIDEO_FEAT_DIM,
        batch_size: int = 16,
        max_frames: int | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model_id = model_id
        self.batch_size = max(1, int(batch_size))
        self.max_frames = max_frames
        self.output_dim = output_dim

        self.model, self.processor = _load_siglip2(model_id, self.device)
        hidden = int(self.model.config.vision_config.hidden_size)

        self.native_feat_dim = hidden
        self.proj = None
        if output_dim is not None and output_dim != hidden:
            self.proj = nn.Linear(hidden, output_dim, bias=False).to(self.device)
            nn.init.orthogonal_(self.proj.weight)

        self.feat_dim = output_dim if output_dim is not None else hidden

    @torch.inference_mode()
    def encode(self, frames: torch.Tensor) -> np.ndarray:
        """
        Encode a lip/video clip.

        Args:
            frames: ``[T, H, W, 3]`` RGB tensor (uint8 or float).

        Returns:
            ``numpy.ndarray`` of shape ``[T, feat_dim]`` (float32).
        """
        if frames.ndim != 4:
            raise ValueError(f"Expected [T, H, W, 3], got {tuple(frames.shape)}")

        if self.max_frames is not None and frames.shape[0] > self.max_frames:
            frames = frames[: self.max_frames]

        images = _frames_to_pil_list(frames)
        embeddings = []

        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            feats = self.model.get_image_features(**inputs)
            if self.proj is not None:
                feats = self.proj(feats.to(dtype=self.proj.weight.dtype))
            embeddings.append(feats.float().cpu())

        out = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
        return out
