# Fast Feature Extraction (SigLIP2 + Qwen2-Audio)

Batched Qwen audio + decord/prefetch SigLIP2 video. Outputs match the baseline extractor.

**Launcher:** one script for everything — `scripts/run_siglip2_extract.sh`  
**Helpers:** `scripts/probe_gpus.sh`, `scripts/diagnose_cuda.sh`, `scripts/activate_env.sh`

| Python module | Role |
|---------------|------|
| `qwen_audio_encoder_batched.py` | Batched Qwen2-Audio encoder |
| `siglip2_video_encoder_fast.py` | decord decode + prefetch |
| `feature_extractor_TVA2_siglip2_fast.py` | Fast pipeline (`FAST=1`, default) |
| `feature_extractor_TVA2_siglip2.py` | Baseline pipeline (`FAST=0`) |

---

## Setup

```bash
cd /path/to/MISP-QEKS
cp config/paths.env.tars.example config/paths.env   # optional on tars
source scripts/activate_env.sh
```

Requires `transformers>=4.49`, `decord` (recommended), CUDA PyTorch. Models cache in `hf_cache/`.

---

## How to run

All examples use **`scripts/run_siglip2_extract.sh`**. Set env vars; no other shell wrappers needed.

### Full train, single GPU (recommended on tars)

```bash
PREFIX=train NOHUP=1 bash scripts/run_siglip2_extract.sh
tail -f results/siglip2_train_fast_gpu0.log   # expect: Using: cuda
```

Resume: re-run the same command (skips finished pairs).

### Smoke test

```bash
MAX_SAMPLES=10 ALLOW_LOCAL=1 bash scripts/run_siglip2_extract.sh
```

### Eval (baseline, slower)

```bash
FAST=0 PREFIX=eval bash scripts/run_siglip2_extract.sh
```

### Multi-GPU (optional)

Only if `bash scripts/probe_gpus.sh` passes for every GPU:

```bash
GPUS=0,1,2,3,4,5 PREFIX=train bash scripts/run_siglip2_extract.sh
```

Restart one failed shard:

```bash
GPU=1 START_INDEX=83334 MAX_SAMPLES=83334 NOHUP=1 bash scripts/run_siglip2_extract.sh
```

### Env vars (launcher)

| Var | Default | Description |
|-----|---------|-------------|
| `FAST` | `1` | `1` = fast path, `0` = baseline |
| `PREFIX` | `train` | `train`, `eval`, `dev`, `eval_unseen` |
| `GPU` | `0` | Single-GPU id |
| `GPUS` | — | Comma list → multi-GPU, e.g. `0,1,2,3,4,5` |
| `MAX_SAMPLES` | `0` | `0` = full split |
| `START_INDEX` | `0` | Shard offset / resume |
| `NOHUP` | `0` | `1` = background |
| `PREFLIGHT` | `1` | Run `probe_gpus.sh` first |
| `REQUIRE_CUDA` | `1` | Exit if CUDA unavailable (fast path) |

Direct Python (advanced): `cd data_prepare && python feature_extractor_TVA2_siglip2_fast.py --prefix eval --allow-local`

---

## After extraction

Full train:

```bash
python scripts/rebuild_shuf_scp.py --prefix train
bash run_train.sh
```

Partial train:

```bash
python scripts/build_partial_train_scp.py --max_pairs 50000
bash run_train_siglip2_quick.sh
```

---

## Output layout

```
features/<prefix>/lip_siglip2/   # video
features/<prefix>/wav/         # clean audio
features/<prefix>/wav_*db/     # noisy audio
npy/<prefix>/*.npy             # pair metadata
data_list/shuf_<split>.scp
```

---

## Troubleshooting

**CUDA / `probe_gpus.sh` fails**

```bash
bash scripts/diagnose_cuda.sh
pkill -f feature_extractor_TVA2_siglip2 || true
sleep 15
bash scripts/probe_gpus.sh 0
```

Log out of SSH and back in if the driver is stuck. Extraction cannot run until probe prints `OK:`.

**`decord` missing** — `pip install decord` (falls back to torchvision).

**CUDA OOM** — `AUDIO_BATCH=6 VIDEO_BATCH=24 bash scripts/run_siglip2_extract.sh`

**Resume** — re-run; existing `.npy` files are skipped.

See [README.md](../README.md) for training and evaluation.
