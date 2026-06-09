# Fast Feature Extraction (SigLIP2 + Qwen2-Audio)

This document describes the **fast** feature-extraction path added on branch  
`cursor/swap-whisper-to-qwen-audio-encoder-a2cd`.

The baseline extractor `feature_extractor_TVA2_siglip2.py` processes samples sequentially and runs **one audio clip per GPU forward pass**. The fast path adds two encoder modules and a pipeline script that batch audio, decode video faster, and prefetch the next clips on CPU threads.

---

## Files

| File | Role |
|------|------|
| `qwen_audio_encoder_batched.py` | Batched Qwen2-Audio encoder (multiple waveforms per forward pass) |
| `siglip2_video_encoder_fast.py` | Fast SigLIP2 video encoder (decord + threaded prefetch) |
| `feature_extractor_TVA2_siglip2_fast.py` | End-to-end pipeline using both modules |
| `../scripts/run_siglip2_extract_fast.sh` | Convenience launcher |

**Original (unchanged):** `feature_extractor_TVA2_siglip2.py`, `qwen_audio_encoder.py`, `siglip2_video_encoder.py`

---

## Changes

### 1. Audio — `qwen_audio_encoder_batched.py`

**Problem:** Each enrollment/query waveform (clean + each SNR) triggered a separate `QwenAudioEncoder.encode()` call → many small GPU kernels, poor GPU utilization.

**Solution:** `BatchedQwenAudioEncoder` pads multiple waveforms into one mel batch and runs a single `audio_tower` forward per chunk.

- `encode_batch(audios)` — list in, list of `(1, T, 1280)` tensors out  
- `encode_many(audios, as_numpy=True)` — returns `(T, 1280)` numpy per clip  
- `encode_pending_audio(encoder, jobs)` — helper for `(save_path, waveform)` jobs; skips paths that already exist  

Default batch size: **8** (`--audio_batch_size`).

### 2. Vision — `siglip2_video_encoder_fast.py`

**Problem:** `torchvision.io.read_video` is slow and synchronous; the GPU often waited on CPU decode and PIL conversion.

**Solution:**

- **`read_video_frames_fast()`** — uses [decord](https://github.com/dmlc/decord) when installed (falls back to torchvision)  
- **`FastSiglip2VideoEncoder`** — same SigLIP2 model as the baseline, larger default frame batch (32)  
- **`VideoPrefetcher`** — `ThreadPoolExecutor` decodes upcoming lip videos on CPU while the GPU encodes the current clip  

Default: **32** frames per SigLIP2 batch (`--video_batch_size`), **4** prefetch threads (`--video_workers`).

### 3. Pipeline — `feature_extractor_TVA2_siglip2_fast.py`

Orchestrates the full TVA pipeline (text G2P + SigLIP2 video + Qwen audio + noisy SNRs):

1. Prefetch next sample’s videos on background threads  
2. Encode enroll + query video with `FastSiglip2VideoEncoder`  
3. Collect **all** pending audio jobs for the sample (clean + 4 SNRs × 2 wavs) and flush in batches  
4. Write the same `.npy` metadata and feature paths as the standard extractor  

Outputs remain compatible with training (`run_train_siglip2_quick.sh`, `train.py`, etc.).

---

## Prerequisites

### Environment

```bash
cd /path/to/MISP-QEKS
source scripts/activate_env.sh   # envs/misp-qeks + config/paths.env
```

Required packages (already in `requirements.txt` on this branch):

- `transformers>=4.49` (SigLIP2 + Qwen2-Audio)  
- `decord` (fast video; optional but recommended)  
- `accelerate`, `Pillow`, CUDA PyTorch  

### Data & models

1. **Hugging Face access** to [Igor97/MISP-QEKS](https://huggingface.co/datasets/Igor97/MISP-QEKS) (gated)  
2. Dataset under `MISP_DATA_ROOT` (see `config/paths.env`):  
   `data/train`, `data/eval_seen`, `data/noise`, …  
3. First run downloads **Qwen2-Audio-7B** (audio tower) and **google/siglip2-base-patch16-224** into `hf_cache/`  

### Paths (`config/paths.env`)

```bash
export MISP_DATA_ROOT="/path/to/MISP-QEKS/data"
export MISP_BASELINE_ROOT="/path/to/MISP-QEKS"
```

On the lab machine (`tars`), defaults in `paths_config.py` point to `/local/scratch/linna/...`.

---

## How to run

### Quick smoke test (10 samples)

```bash
source scripts/activate_env.sh
MAX_SAMPLES=10 ALLOW_LOCAL=1 bash scripts/run_siglip2_extract_fast.sh
```

Logs: `results/siglip2_eval_fast.log`

### Full eval split (local / non-tars)

```bash
source scripts/activate_env.sh
cd data_prepare

python feature_extractor_TVA2_siglip2_fast.py \
  --prefix eval \
  --allow-local \
  --audio_batch_size 8 \
  --video_batch_size 32 \
  --video_workers 4
```

### Train split

```bash
python feature_extractor_TVA2_siglip2_fast.py \
  --prefix train \
  --allow-local \
  --audio_batch_size 8 \
  --video_batch_size 32 \
  --video_workers 4
```

`--prefix` choices: `train`, `dev`, `eval`, `eval_unseen` (see `paths_config.PREFIX_CONFIG`).

### CLI options (fast extractor)

| Flag | Default | Description |
|------|---------|-------------|
| `--prefix` | `eval` | Dataset split |
| `--audio_batch_size` | `8` | Qwen waveforms per GPU batch |
| `--video_batch_size` | `32` | SigLIP2 frames per GPU batch |
| `--video_workers` | `4` | CPU threads for video prefetch |
| `--max_samples` | `0` | Limit samples (`0` = all) |
| `--start_index` | `0` | Resume / shard offset |
| `--max_frames` | `50` | Max lip frames per clip |
| `--allow-local` | off | Allow run outside `tars` scratch |
| `--no_rebuild_scp` | off | Skip `shuf_*.scp` rebuild (use for multi-GPU shards) |
| `--no_log_file` | off | Console only |

### Multi-GPU (manual sharding)

Launch one process per GPU with disjoint ranges (same idea as `scripts/run_siglip2_extract_multi_gpu.sh`, but call the **fast** script):

```bash
CUDA_VISIBLE_DEVICES=0 python feature_extractor_TVA2_siglip2_fast.py \
  --prefix train --start_index 0 --max_samples 125000 \
  --no_rebuild_scp --allow-local &

CUDA_VISIBLE_DEVICES=1 python feature_extractor_TVA2_siglip2_fast.py \
  --prefix train --start_index 125000 --max_samples 125000 \
  --no_rebuild_scp --allow-local &
```

When all shards finish, rebuild the training list:

```bash
python scripts/build_partial_train_scp.py --max_pairs 50000   # or full count
```

---

## Output layout

Same as the standard SigLIP2 extractor:

```
features/<prefix>/
  lip_siglip2/          # video features (.npy per clip)
  wav/                  # clean audio features
  wav_5db/ wav_0db/ ... # noisy audio features

npy/<prefix>/           # per-pair metadata dicts
  <anc>+<com>.npy
  <anc>+<com>_5db.npy
  ...

data_list/shuf_<split>.scp   # rebuilt at end (unless --no_rebuild_scp)
```

Video features use folder `lip_siglip2/` (not the legacy CNN `lip/` path).

---

## Baseline vs fast

| | `feature_extractor_TVA2_siglip2.py` | `feature_extractor_TVA2_siglip2_fast.py` |
|---|--------------------------------------|------------------------------------------|
| Audio | 1 waveform / forward | Batched (default 8) |
| Video decode | torchvision, sync | decord + prefetch threads |
| SigLIP2 frame batch | 16 | 32 (configurable) |
| Output format | identical | identical |

Use the **fast** script for large-scale offline extraction; use the **baseline** script if you need to match an older log line-by-line.

---

## Troubleshooting

**`ERROR: This job must run on tars scratch`**  
Pass `--allow-local` or set `MISP_BASELINE_ROOT` / `MISP_DATA_ROOT` to scratch paths on `tars`.

**`decord` import fails**  
Install: `pip install decord` — code falls back to torchvision (slower).

**CUDA OOM**  
Lower `--audio_batch_size` and/or `--video_batch_size`.

**Resume after interrupt**  
Re-run the same command; existing `.npy` feature files are skipped. Use `--start_index` to skip whole sample ranges for sharded jobs.

**HF gated dataset**  
`huggingface-cli login` after access is approved.

---

## Next steps after extraction

```bash
bash run_train_siglip2_quick.sh    # quick train on partial scp
# or
bash scripts/run_train_local.sh    # full training (update datalist paths)
```

See the main [README.md](../README.md) for evaluation and baseline metrics.
