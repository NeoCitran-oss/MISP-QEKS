"""Batched Qwen3-Omni AuT encoder for offline feature extraction."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, List, Sequence, Union

import numpy as np
import torch

from qwen3_audio_encoder import DEFAULT_MODEL_ID, MelBatch, Qwen3AudioEncoder

ArrayLike = Union[np.ndarray, torch.Tensor]


def _normalize_waveform(audio: ArrayLike) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max + 1)
        audio = audio.astype(np.float32) / max_val
    return audio.astype(np.float32)


class BatchedQwen3AudioEncoder(Qwen3AudioEncoder):
    """Qwen3 AuT encoder; ``batch_size`` controls how many clips per mel/GPU step."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device=None,
        max_frames: int = 100,
        dtype=None,
        cache_dir=None,
        batch_size: int = 32,
        llm_proj: bool = False,
        mel_workers: int = 2,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            max_frames=max_frames,
            dtype=dtype,
            cache_dir=cache_dir,
            llm_proj=llm_proj,
        )
        self.batch_size = max(1, int(batch_size))
        self.mel_workers = max(0, int(mel_workers))
        self._mel_pool: ThreadPoolExecutor | None = None
        if self.mel_workers > 0:
            self._mel_pool = ThreadPoolExecutor(max_workers=self.mel_workers)

    def shutdown(self) -> None:
        if self._mel_pool is not None:
            self._mel_pool.shutdown(wait=True)
            self._mel_pool = None

    @torch.inference_mode()
    def encode_batch_tensors(
        self,
        waveforms: Sequence[np.ndarray],
    ) -> List[torch.Tensor]:
        if not waveforms:
            return []

        normalized = [_normalize_waveform(a) for a in waveforms]
        sub_batches = [
            normalized[start : start + self.batch_size]
            for start in range(0, len(normalized), self.batch_size)
        ]
        if len(sub_batches) <= 1 or self._mel_pool is None:
            outputs: List[torch.Tensor] = []
            for sub in sub_batches:
                outputs.extend(self.encode_from_mel(self.prepare_mel_batch(sub)))
            return outputs

        outputs: List[torch.Tensor] = []
        pending: Future[MelBatch] = self._mel_pool.submit(self.prepare_mel_batch, sub_batches[0])
        for i, sub in enumerate(sub_batches):
            mel = pending.result()
            if i + 1 < len(sub_batches):
                pending = self._mel_pool.submit(self.prepare_mel_batch, sub_batches[i + 1])
            outputs.extend(self.encode_from_mel(mel))
        return outputs

    @torch.inference_mode()
    def encode_batch(self, audios: Sequence[ArrayLike]) -> List[torch.Tensor]:
        if not audios:
            return []
        normalized = [_normalize_waveform(a) for a in audios]
        return self.encode_batch_tensors(normalized)

    @torch.inference_mode()
    def encode_many(
        self,
        audios: Sequence[ArrayLike],
        *,
        as_numpy: bool = True,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        tensors = self.encode_batch(audios)
        if not as_numpy:
            return tensors
        return [t.squeeze(0).numpy() for t in tensors]


def encode_audio_jobs(
    encoder: BatchedQwen3AudioEncoder,
    jobs: Sequence[tuple[str, ArrayLike]],
    *,
    as_numpy: bool = True,
) -> dict[str, Union[np.ndarray, torch.Tensor]]:
    if not jobs:
        return {}
    paths = [p for p, _ in jobs]
    waveforms = [w for _, w in jobs]
    feats = encoder.encode_many(waveforms, as_numpy=as_numpy)
    return dict(zip(paths, feats))
