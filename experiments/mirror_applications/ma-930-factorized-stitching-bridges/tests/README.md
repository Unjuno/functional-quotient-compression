Verification performed:

- `python -m py_compile experiments/mirror_applications/ma-930-factorized-stitching-bridges/source/run_screen.py`
- Full frozen CPU screen completed for all 3 seeds and 48 held-out pair/layer rows.
- Factorized connector state reload replay max difference was 0.
- Native target-tail identity was checked on the 8M model at layers 2/4/6/8; max native-logit difference was 0.
- Training/validation source files and model revisions/hashes are recorded in `source/data_manifest.json`.

The failed pre-result run that accumulated full vocabulary logits did not emit metrics and is documented in `AMENDMENTS.md`.
