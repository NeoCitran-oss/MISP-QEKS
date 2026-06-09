"""Fast SigLIP 2 video encoder: decord decode + threaded prefetch + frame batching."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from siglip2_video_encoder import Siglip2VideoEncoder


def read_video_frames_fast(path: str, max_frames: int | None = None) -> torch.Tensor:
    """Load ``[T, H, W, 3]`` uint8 tensor. Uses decord when available."""
    if not os.path.isfile(path):
        alt = path.replace(".mp4", ".m4p")
        if os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(path)

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        if max_frames is not None and n > max_frames:
            n = max_frames
        frames = vr.get_batch(range(n)).asnumpy()
        return torch.from_numpy(frames)
    except Exception:
        import torchvision

        frames, _, _ = torchvision.io.read_video(path, pts_unit="sec")
        if max_frames is not None and frames.shape[0] > max_frames:
            frames = frames[:max_frames]
        return frames


def _frames_to_pil_batch(frames: torch.Tensor) -> list[Image.Image]:
    """Vectorized uint8 conversion then PIL list (still needed for HF processor)."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {tuple(frames.shape)}")

    arr = frames.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return [Image.fromarray(arr[t], mode="RGB") for t in range(arr.shape[0])]


class FastSiglip2VideoEncoder(Siglip2VideoEncoder):
    """SigLIP 2 with decord IO and optional background video prefetch."""

    def read_frames(self, path: str) -> torch.Tensor:
        return read_video_frames_fast(path, max_frames=self.max_frames)

    @torch.inference_mode()
    def encode_path(self, video_path: str) -> np.ndarray:
        return self.encode(self.read_frames(video_path))

    @torch.inference_mode()
    def encode(self, frames: torch.Tensor) -> np.ndarray:
        if frames.ndim != 4:
            raise ValueError(f"Expected [T, H, W, 3], got {tuple(frames.shape)}")

        if self.max_frames is not None and frames.shape[0] > self.max_frames:
            frames = frames[: self.max_frames]

        images = _frames_to_pil_batch(frames)
        embeddings = []

        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            feats = self.model.get_image_features(**inputs)
            if self.proj is not None:
                feats = self.proj(feats.to(dtype=self.proj.weight.dtype))
            embeddings.append(feats.float().cpu())

        return torch.cat(embeddings, dim=0).numpy().astype(np.float32)


class VideoPrefetcher:
    """Prefetch video decode on CPU threads while GPU encodes the current clip."""

    def __init__(self, max_workers: int = 4, max_frames: int | None = 50):
        self.max_workers = max(1, int(max_workers))
        self.max_frames = max_frames
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._futures: Dict[str, Future] = {}

    def schedule(self, key: str, path: str) -> None:
        if key in self._futures:
            return
        self._futures[key] = self._pool.submit(
            read_video_frames_fast, path, self.max_frames
        )

    def get(self, key: str) -> torch.Tensor:
        if key not in self._futures:
            raise KeyError(key)
        return self._futures.pop(key).result()

    def discard(self, key: str) -> None:
        """Drop a scheduled decode (e.g. feature already on disk). Frees thread + RAM."""
        fut = self._futures.pop(key, None)
        if fut is None:
            return
        if fut.done():
            try:
                fut.result()
            except Exception:
                pass
        else:
            fut.cancel()

    def shutdown(self, wait: bool = True) -> None:
        for key in list(self._futures):
            self.discard(key)
        self._pool.shutdown(wait=wait)
