"""Qwen2-Audio audio-encoder backbone.

This module rips out *only* the audio encoder (audio tower) of Qwen2-Audio and
uses it as a stand-alone speech feature extractor. The Qwen large language model
(text decoder) and the audio->text multimodal projector are dropped, so nothing
but the acoustic encoder is kept in memory.

The encoder is a Whisper-large-v3 style transformer followed by an average
pooling layer, producing one frame every 40 ms (25 fps) with a hidden size of
``QWEN_AUDIO_FEAT_DIM`` (1280). This replaces the previous Whisper-tiny encoder
(384-dim, 50 fps).

Two loading strategies are supported:
  1. Lightweight (default): download/read only the checkpoint shards that hold
     ``audio_tower`` weights and load them into a freshly built encoder. The 7B
     language model is never allocated.
  2. Fallback: load the full ``Qwen2AudioForConditionalGeneration`` and keep only
     its audio tower (works on any transformers version, but heavier).
"""

import glob
import json
import os

import numpy as np
import torch

# Hidden size of the Qwen2-Audio audio encoder output (whisper-large-v3 width).
QWEN_AUDIO_FEAT_DIM = 1280

# The Qwen2-Audio encoder applies an extra AvgPool1d(2) after the transformer,
# so it emits one frame per 640 input samples (40 ms @ 16 kHz / 25 fps).
QWEN_AUDIO_FRAME_STRIDE = 640

DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B"
SAMPLE_RATE = 16000

# Substring identifying audio-tower weights inside the full checkpoint. Works for
# both the flat layout (``audio_tower.*``) and the nested v5 layout
# (``model.audio_tower.*``).
_AUDIO_TOWER_KEY = "audio_tower."


def _build_audio_encoder(audio_config):
    """Instantiate a bare ``Qwen2AudioEncoder`` from an audio sub-config."""
    from transformers import AutoModel

    return AutoModel.from_config(audio_config)


def _load_audio_tower_lightweight(model_id, dtype, cache_dir=None):
    """Build the audio encoder and load *only* its weights from the checkpoint.

    Downloads/reads just the shards containing ``audio_tower`` weights so the
    multi-billion parameter language model is never materialized.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    audio_config = getattr(config, "audio_config", config)
    encoder = _build_audio_encoder(audio_config)

    local_files = _resolve_audio_shards(model_id, cache_dir=cache_dir)

    state = {}
    for shard_path in local_files:
        shard = load_file(shard_path)
        for key, tensor in shard.items():
            if _AUDIO_TOWER_KEY in key:
                stripped = key.split(_AUDIO_TOWER_KEY, 1)[1]
                state[stripped] = tensor

    if not state:
        raise RuntimeError("No audio_tower weights found in checkpoint shards.")

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    # Position-id style buffers are allowed to be missing; anything else is a bug.
    real_missing = [k for k in missing if "position" not in k]
    if real_missing:
        raise RuntimeError(f"Missing audio_tower weights: {real_missing[:8]} ...")

    if dtype is not None:
        encoder = encoder.to(dtype)
    return encoder


def _resolve_audio_shards(model_id, cache_dir=None):
    """Return local paths of the safetensors shards holding audio_tower weights."""
    # Local directory checkpoint.
    if os.path.isdir(model_id):
        index_path = os.path.join(model_id, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted({fname for key, fname in weight_map.items() if _AUDIO_TOWER_KEY in key})
            return [os.path.join(model_id, s) for s in shards]
        return sorted(glob.glob(os.path.join(model_id, "*.safetensors")))

    # Remote hub checkpoint: fetch the index, pick only the needed shards.
    from huggingface_hub import hf_hub_download

    try:
        index_path = hf_hub_download(
            model_id, "model.safetensors.index.json", cache_dir=cache_dir
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({fname for key, fname in weight_map.items() if _AUDIO_TOWER_KEY in key})
        return [
            hf_hub_download(model_id, s, cache_dir=cache_dir) for s in shards
        ]
    except Exception:
        # Single-file checkpoint (not sharded): download the whole safetensors.
        return [hf_hub_download(model_id, "model.safetensors", cache_dir=cache_dir)]


def _load_audio_tower_full(model_id, dtype, cache_dir=None):
    """Fallback: load the full model and keep only the audio tower."""
    from transformers import Qwen2AudioForConditionalGeneration

    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype is not None else torch.float32,
        cache_dir=cache_dir,
    )

    # transformers >= 5 nests submodules under ``.model``; older versions are flat.
    holder = full_model.model if hasattr(full_model, "model") and hasattr(full_model.model, "audio_tower") else full_model
    audio_tower = holder.audio_tower

    # Detach from the parent so the LLM/projector can be garbage collected.
    if hasattr(holder, "language_model"):
        holder.language_model = None
    if hasattr(holder, "multi_modal_projector"):
        holder.multi_modal_projector = None
    del full_model
    return audio_tower


class QwenAudioEncoder:
    """Wraps the Qwen2-Audio audio tower as a frozen speech feature extractor.

    Example
    -------
    >>> enc = QwenAudioEncoder(device="cuda")
    >>> feats = enc(waveform)          # waveform: float32 mono 16 kHz
    >>> feats.shape                    # (1, <=max_frames, 1280)
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID, device=None, max_frames=100,
                 dtype=None, lightweight=True, cache_dir=None):
        from transformers import AutoProcessor

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_frames = max_frames
        self.cache_dir = cache_dir

        audio_tower = None
        if lightweight:
            try:
                audio_tower = _load_audio_tower_lightweight(
                    model_id, dtype, cache_dir=cache_dir
                )
            except Exception as exc:  # pragma: no cover - fall back to full load
                print(f"[QwenAudioEncoder] lightweight load failed ({exc}); "
                      f"falling back to full-model load.")
        if audio_tower is None:
            audio_tower = _load_audio_tower_full(model_id, dtype, cache_dir=cache_dir)

        self.audio_tower = audio_tower.to(self.device).eval()
        for param in self.audio_tower.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.feature_extractor = self.processor.feature_extractor

    @torch.no_grad()
    def __call__(self, audio):
        return self.encode(audio)

    @torch.no_grad()
    def encode(self, audio):
        """Encode a mono 16 kHz waveform into Qwen audio features.

        Parameters
        ----------
        audio : np.ndarray | torch.Tensor
            1-D waveform, float32 in [-1, 1] (or int16 PCM, which is normalized).

        Returns
        -------
        torch.Tensor of shape ``(1, T, QWEN_AUDIO_FEAT_DIM)`` on CPU, where
        ``T <= max_frames``.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio)

        # Normalize integer PCM to float32 in [-1, 1].
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max + 1)
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device, dtype=self.audio_tower.dtype)

        encoder_out = self.audio_tower(input_features)
        audio_embed = encoder_out.last_hidden_state  # (1, T_enc, 1280)

        audio_embed = audio_embed[:, : self.max_frames, :].float().cpu()
        return audio_embed
