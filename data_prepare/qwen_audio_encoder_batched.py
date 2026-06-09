"""Batched Qwen2-Audio encoder for faster offline feature extraction.

Wraps :class:`QwenAudioEncoder` and runs multiple waveforms per GPU forward pass
(padded mel batch). Use via :meth:`encode_many` or :meth:`encode_batch`.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Union

import numpy as np
import torch
from qwen_audio_encoder import (
    QWEN_AUDIO_FEAT_DIM,
    QWEN_AUDIO_FRAME_STRIDE,
    DEFAULT_MODEL_ID,
    QwenAudioEncoder,
    SAMPLE_RATE,
)

# Qwen2-Audio / Whisper-style mel context (30 s @ 16 kHz).
QWEN_MEL_MAX_FRAMES = 3000

ArrayLike = Union[np.ndarray, torch.Tensor]


def _normalize_waveform(audio: ArrayLike) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max + 1)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    return audio


def _estimate_output_frames(num_samples: int, max_frames: int) -> int:
    if num_samples <= 0:
        return 0
    frames = math.ceil(num_samples / QWEN_AUDIO_FRAME_STRIDE)
    return min(max_frames, frames)


class BatchedQwenAudioEncoder(QwenAudioEncoder):
    """Qwen audio tower with cross-clip batching."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device=None,
        max_frames: int = 100,
        dtype=None,
        lightweight: bool = True,
        cache_dir=None,
        batch_size: int = 8,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            max_frames=max_frames,
            dtype=dtype,
            lightweight=lightweight,
            cache_dir=cache_dir,
        )
        self.batch_size = max(1, int(batch_size))
        self.mel_max_frames = int(
            getattr(self.feature_extractor, "nb_max_frames", QWEN_MEL_MAX_FRAMES)
        )

    def _pad_mel_to_model_length(self, input_features: torch.Tensor) -> torch.Tensor:
        """Qwen2-Audio expects fixed-length mel (3000 frames), not batch-local padding."""
        target = self.mel_max_frames
        t = input_features.shape[-1]
        if t < target:
            input_features = torch.nn.functional.pad(
                input_features, (0, target - t), mode="constant", value=0.0
            )
        elif t > target:
            input_features = input_features[..., :target]
        return input_features

    @torch.no_grad()
    def encode(self, audio: ArrayLike) -> torch.Tensor:
        """Single clip (same API as base class)."""
        return self.encode_batch([audio])[0]

    @torch.no_grad()
    def encode_batch(self, audios: Sequence[ArrayLike]) -> List[torch.Tensor]:
        """Encode a list of waveforms; returns CPU tensors ``(1, T, 1280)`` each."""
        if not audios:
            return []
        if len(audios) == 1:
            return [self._encode_single(audios[0])]

        normalized = [_normalize_waveform(a) for a in audios]
        lengths = [len(a) for a in normalized]
        outputs: List[torch.Tensor] = []

        for start in range(0, len(normalized), self.batch_size):
            chunk = normalized[start : start + self.batch_size]
            chunk_lens = lengths[start : start + self.batch_size]
            outputs.extend(self._forward_chunk(chunk, chunk_lens))
        return outputs

    @torch.no_grad()
    def encode_many(
        self,
        audios: Sequence[ArrayLike],
        *,
        as_numpy: bool = True,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """Encode waveforms; default return is ``numpy (T, 1280)`` per clip."""
        tensors = self.encode_batch(audios)
        if not as_numpy:
            return tensors
        out = []
        for t in tensors:
            arr = t.squeeze(0).numpy()
            out.append(arr)
        return out

    @torch.no_grad()
    def _encode_single(self, audio: ArrayLike) -> torch.Tensor:
        wav = _normalize_waveform(audio)
        inputs = self.feature_extractor(
            wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(
            self.device, dtype=self.audio_tower.dtype
        )
        input_features = self._pad_mel_to_model_length(input_features)
        encoder_out = self.audio_tower(input_features)
        n_frames = _estimate_output_frames(len(wav), self.max_frames)
        return encoder_out.last_hidden_state[:, :n_frames, :].float().cpu()

    @torch.no_grad()
    def _forward_chunk(
        self, waveforms: List[np.ndarray], sample_lengths: List[int]
    ) -> List[torch.Tensor]:
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(
            self.device, dtype=self.audio_tower.dtype
        )
        input_features = self._pad_mel_to_model_length(input_features)
        encoder_out = self.audio_tower(input_features)
        hidden = encoder_out.last_hidden_state.float().cpu()

        results: List[torch.Tensor] = []
        for i, n_samples in enumerate(sample_lengths):
            n_frames = _estimate_output_frames(n_samples, self.max_frames)
            results.append(hidden[i : i + 1, :n_frames, :])
        return results


def encode_pending_audio(
    encoder: BatchedQwenAudioEncoder,
    jobs: Iterable[tuple[str, ArrayLike]],
    *,
    as_numpy: bool = True,
) -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """Run batched encode for ``(save_path, waveform)`` jobs; skip existing paths."""
    import os

    pending: list[tuple[str, ArrayLike]] = []
    done: dict[str, Union[np.ndarray, torch.Tensor]] = {}

    for path, wav in jobs:
        if os.path.exists(path):
            continue
        pending.append((path, wav))

    if not pending:
        return done

    waveforms = [w for _, w in pending]
    feats = encoder.encode_many(waveforms, as_numpy=as_numpy)
    for (path, _), feat in zip(pending, feats):
        done[path] = feat
    return done
