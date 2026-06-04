# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

Official PyTorch baseline for **MISP-QEKS** (text–audio–visual Query-by-Example Keyword Spotting). There is no web server or Docker stack: workflows are batch **train** / **test** scripts over preprocessed `.npy` features and forced-alignment TextGrids.

### Python environment

- Use the project venv: `source /workspace/.venv/bin/activate` (or prefix commands with `.venv/bin/`).
- First-time VM images may need `sudo apt install -y python3.12-venv` before `python3 -m venv .venv` works.
- `requirements.txt` does not list **`praatio`** (required by `model/fadata.py` and dataloaders); install it alongside requirements (the update script does this).

### Hardware expectations

- **CUDA GPU is required** for `train.py`, `test.py`, and model construction (`TVA_KWS_PLCL_AVmask` calls `.cuda()` in `__init__`).
- `run_train.sh`, `run_test.sh`, and `run_debug.sh` source a hard-coded author path (`/home3/asrkws/.../bashrc_multimodal_kws`). Prefer invoking Python directly with flags instead of those shells unless you replicate that environment.

### Data and paths

- Download the dataset from [Hugging Face: Igor97/MISP-QEKS](https://huggingface.co/datasets/Igor97/MISP-QEKS) into `data/` (`train/`, `dev/`, `eval_seen/`, `eval_unseen/`).
- `dataset_list/shuf_*.scp` files use placeholder roots like `/my_path/npy/...`; repoint `--datalist_dir` and/or edit `.scp` paths to your feature store.
- Training/eval also needs `fa_data/*.TextGrid` paths derived from wav paths inside each `.npy` (see `loader/dataloader.py`).
- Eval checkpoints load as `{model_path}epoch{N}.pth` (see `test.py`); the README `--ckpt` flag is not wired in `test.py`.

### Commands (from repo root, with venv active)

| Task | Command |
|------|---------|
| Syntax check (no linter configured) | `python -m compileall -q .` |
| Train | `CUDA_VISIBLE_DEVICES=0 python train.py --datalist_dir <path> --train_csv train --eval_csv eval_inset,eval_outset --out_dir ./train/model/ ...` |
| Eval | `CUDA_VISIBLE_DEVICES=0 python test.py --datalist_dir <path> --model_path ./train/model/ --bgn_epoch 9 --end_epoch 9 ...` |
| CLI help | `python train.py --help` / `python test.py --help` |

There is no pytest suite; integration testing is `test.py` on GPU with real features and a checkpoint.

### CPU-only smoke check (no dataset)

When no GPU or data are available, verify imports and evaluation metrics:

```bash
.venv/bin/python -c "from test import compute_eer; import numpy as np; from sklearn.metrics import roc_auc_score; \
l=np.array([0,1,0,1]); s=np.array([0.2,0.8,0.3,0.7]); print('EER', compute_eer(l,s), 'AUC', roc_auc_score(l,s))"
```
