# nanoGPT research baseline

This directory contains the compact nanoGPT baseline used for Mirror-application A/B experiments.

- provenance and omissions: `VENDOR_NOTES.md`
- license: `LICENSE`
- core model: `model.py`
- training loop: `train.py`
- sampling: `sample.py`
- benchmark: `bench.py`

Do not implement Mirror experiments directly in this directory. Keep this baseline stable and put variants under `experiments/mirror_applications/` or thin project adapters.

The user-supplied original archive is not copied into Git history as a binary ZIP; its SHA-256 is recorded in `VENDOR_NOTES.md`.