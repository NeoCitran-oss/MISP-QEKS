"""Qwen3-Omni AuT audio encoder (lightweight: one safetensors shard only).

Uses the Thinker's ``audio_tower`` from ``Qwen/Qwen3-Omni-30B-A3B-Instruct``.
By default returns encoder features at ``d_model=1280`` (before the LLM projection
to 2048) so they remain compatible with the existing TVA model.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLE_RATE = 16000

# AuT encoder width (matches Qwen2-Audio tower width used downstream).
QWEN3_AUDIO_FEAT_DIM = 1280
# 12.5 Hz token rate -> 16000 / 12.5 = 1280 samples per frame.
QWEN3_AUDIO_FRAME_STRIDE = 1280

_AUDIO_TOWER_PREFIX = "thinker.audio_tower."


@dataclass
class MelBatch:
    """CPU-side mel features ready for GPU encode."""

    input_features: torch.Tensor
    feature_attention_mask: torch.Tensor


def _resolve_audio_shards(model_id: str, cache_dir=None) -> List[str]:
    if os.path.isdir(model_id):
        index_path = os.path.join(model_id, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted(
                {fname for key, fname in weight_map.items() if _AUDIO_TOWER_PREFIX in key}
            )
            return [os.path.join(model_id, s) for s in shards]
        return sorted(glob.glob(os.path.join(model_id, "*.safetensors")))

    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=cache_dir)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({fname for key, fname in weight_map.items() if _AUDIO_TOWER_PREFIX in key})
    return [hf_hub_download(model_id, s, cache_dir=cache_dir) for s in shards]


def _load_audio_tower(model_id: str, dtype, cache_dir=None):
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    audio_config = config.thinker_config.audio_config
    encoder = Qwen3OmniMoeAudioEncoder(audio_config)

    state = {}
    for shard_path in _resolve_audio_shards(model_id, cache_dir=cache_dir):
        shard = load_file(shard_path)
        for key, tensor in shard.items():
            if key.startswith(_AUDIO_TOWER_PREFIX):
                state[key[len(_AUDIO_TOWER_PREFIX) :]] = tensor

    if not state:
        raise RuntimeError(f"No {_AUDIO_TOWER_PREFIX} weights found for {model_id}")

    missing, _unexpected = encoder.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "position" not in k]
    if real_missing:
        raise RuntimeError(f"Missing Qwen3 audio_tower weights: {real_missing[:8]} ...")

    if dtype is not None:
        encoder = encoder.to(dtype)
    return encoder


class Qwen3AudioEncoder:
    """Frozen Qwen3-Omni AuT encoder for offline feature extraction."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device=None,
        max_frames: int = 100,
        dtype=None,
        cache_dir=None,
        llm_proj: bool = False,
    ):
        from transformers import AutoFeatureExtractor

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_id = model_id
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        self.llm_proj = llm_proj
        self.feat_dim = 2048 if llm_proj else QWEN3_AUDIO_FEAT_DIM

        if dtype is None and self.device.type == "cuda":
            dtype = torch.float16

        self.audio_tower = _load_audio_tower(model_id, dtype, cache_dir=cache_dir)
        self.audio_tower = self.audio_tower.to(self.device).eval()
        for param in self.audio_tower.parameters():
            param.requires_grad = False

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _forward_packed(
        self,
        packed_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            _get_feat_extract_output_lengths as mel_conv_out_len,
        )

        # Run the tower but stop before LLM projection unless requested.
        tower: Qwen3OmniMoeAudioEncoder = self.audio_tower
        aftercnn_lens = mel_conv_out_len(feature_lens)
        chunk_num = torch.ceil(feature_lens / (tower.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [tower.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (tower.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = tower.n_window * 2

        chunk_list = packed_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = mel_conv_out_len(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
            padded_embed = torch.nn.functional.gelu(tower.conv2d1(chunk))
            padded_embed = torch.nn.functional.gelu(tower.conv2d2(padded_embed))
            padded_embed = torch.nn.functional.gelu(tower.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = tower.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        positional_embedding = (
            tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (tower.n_window_infer // (tower.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in tower.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]

        hidden_states = tower.ln_post(hidden_states)
        if self.llm_proj:
            hidden_states = tower.proj1(hidden_states)
            hidden_states = tower.act(hidden_states)
            hidden_states = tower.proj2(hidden_states)
        return hidden_states

    def prepare_mel_batch(self, waveforms: Sequence[np.ndarray]) -> MelBatch:
        """CPU mel extraction (HF feature extractor); safe to run off the main thread."""
        inputs = self.feature_extractor(
            list(waveforms),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return MelBatch(inputs.input_features, inputs.attention_mask.long())

    @torch.inference_mode()
    def encode_from_mel(self, mel: MelBatch) -> List[torch.Tensor]:
        """GPU forward for a precomputed mel batch."""
        input_features = mel.input_features.to(
            self.device, dtype=self.audio_tower.dtype, non_blocking=True
        )
        feature_attention_mask = mel.feature_attention_mask.to(self.device, non_blocking=True)
        feature_lens = feature_attention_mask.sum(-1).long()

        packed = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = self._forward_packed(packed, feature_lens)
        else:
            hidden = self._forward_packed(packed, feature_lens)

        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            _get_feat_extract_output_lengths,
        )

        out_lens = _get_feat_extract_output_lengths(feature_lens.detach().cpu())
        hidden = hidden.float().cpu()
        outputs: List[torch.Tensor] = []
        offset = 0
        for raw_n in out_lens.tolist():
            n = min(int(raw_n), self.max_frames)
            outputs.append(hidden[offset : offset + n].unsqueeze(0))
            offset += int(raw_n)
        return outputs

    @torch.inference_mode()
    def encode_batch_tensors(
        self,
        waveforms: Sequence[np.ndarray],
    ) -> List[torch.Tensor]:
        if not waveforms:
            return []
        mel = self.prepare_mel_batch(waveforms)
        return self.encode_from_mel(mel)

    @torch.inference_mode()
    def encode_many(
        self,
        waveforms: Sequence[np.ndarray],
        *,
        as_numpy: bool = True,
    ) -> List:
        tensors = self.encode_batch_tensors(waveforms)
        if not as_numpy:
            return tensors
        return [t.squeeze(0).numpy() for t in tensors]
