# A13 — Evolution Strategies for two-dimensional held-out task codes

**Status (2026-11-15): exploratory CPU synthetic research. NOT a general Mirror-specific compression or Transformer-capacity result.**

## Result

On the exact **A12 frozen meta-trained scaffolds** (tent/sine, three worlds × two training seeds × Mirror/Shift/FiLM), an elite-selection diagonal Gaussian ES (P32, G30, 960 support-fitness candidate evaluations/task) escapes many poor 2D code optimizations. No teacher function coefficients enter search: 32 labeled support pairs/task are the *only* fitness data.

| Tent model | A13 Adam IID NRMSE | A13 ES IID/OOD | Joint old/IID/OOD pass | Stored FP32 bytes |
|---|---:|---:|---:|---:|
| Mirror | 0.732 | 0.335 / 0.499 | 0/3 worlds | 1004 |
| Shift | 0.754 | **0.089 / 0.104** | **3/3** | 1020 |
| FiLM | 0.676 | **0.142 / 0.156** | **3/3** | 1016 |

After this discovery, **A13c** learned **36 new shared scaffolds from scratch** on worlds 1300–1302 and repeated the frozen search protocol: tent Shift ES 3/3, FiLM ES 2/3, Mirror ES 0/3; sine 0/3 for every method. It is *post-discovery replication*, not fully independent confirmatory design.

**A13b** later compared equal-candidate-evaluation Sobol and four-restart Adam. Both were weaker than ES on tent Shift/FiLM in the tested conditions. **A13d** gave Mirror 10× ES search evaluations and still found zero joint quality-passing worlds across six worlds. None of the above proves a global minimum or Mirror impossibility.

**Posthoc learned-code interpretation:** within each parent, a 2D ES Shift/FiLM code predicts the triangle's latent center and width from eight task codes with high LOOCV R². At the *new A13c worlds*, Shift median R² was 0.9998 (center), 0.9768 (width); FiLM 0.9974/0.9013; Mirror 0.7493/−1.5223. The teacher parameters were read only for this *after-training interpretation*, never for optimizer selection. This is an important potential path toward amortizing ES into a small task-to-code encoder.

**Strict boundary:** only Shift/FiLM achieve the quality gate; the ES result is **code optimization**, not a Mirror-specific compression advantage. Most of the Adam deficiency is extreme per-task failures rather than uniformly worse median support fit. Sine has an additional OOD/generalization failure.

## Reproduce

Baseline sources are sibling `../A11/a11.py` and `../A12/a12.py` (also copied into the evidence ZIP for standalone verification). Main A13 depends on the preceding A12 evidence ZIP, SHA256 `0c3ee59af6bc4ccbab569f65ec16a38cd4b2a85ff2fb52b8ed00e5f0da4adc94`.

```bash
# Python 3.13; Linux CPU PyTorch 2.10.0+cpu, NumPy 2.3.5,
# SciPy 1.17, pandas 2.2.3, scikit-learn 1.8, pytest 9
python -m pytest -q test_a13.py test_audit.py test_followup.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --out main --a12-zip /path/to/mirror_capacity_a12_20261115.zip --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py --out main

# Post-discovery studies require the main locked run:
python followup.py --main main --out followup --workers 4
python audit_followup.py --main main --out followup
python replicate_repo.py --out replication --workers 4
python audit_replication.py --out replication
python es10.py --out es10 --main main --replication replication
python audit_es10.py
```

Run in *new empty output directories*. Audit stages reject previously existing audit outputs and compare whole-set checkpoint SHA-256/size and byte-identical deserialization **before** opening IID/OOD labels. The four-stage reproduction includes meta-training 2400 updates per fresh-world parent; it takes more training work than A13 adaptation alone. The original `replicate.py` is kept byte-for-byte for evidence integrity; `replicate_repo.py` loads the sibling A12 module in a normal GitHub checkout.

## Data and forensic checks

- `main/`: A13 main **36 copied verified parents +108 final model files**, 288 per-task ES policy trajectories and diagnostics, 324 split scores, SHA locks, seed-paired world results.
- `followup/`: A13b **72 models** (4-restart Adam and Sobol), 216 split scores, SHA locks.
- `replication/`: A13c **36 independently trained parents +108 models**, 324 split scores; byte-identical replays on two new parent conditions.
- `es10/`: A13d **12 models**, 36 split scores; 10× ES search for Mirror only, no quality passes.
- `main/policy_example_tent_shift_w1300_seed0.csv`: sample eight evolved 2D codes with **truth values labeled audit-only**.
- `REPORT_JA.md`: complete derivation, five-way controls, per-family results, variables, uncertainties, limitations, failure mechanisms.
- `EVIDENCE.json` and `MANIFEST_SHA256.txt`: content-addressable, auditable provenance; full artifacts ZIP stores the exact weight files outside ordinary Git history.

Models use A11's exact serialized format: **32-byte little-endian header + all FP32 weights + 32-byte SHA-256 trailer**. These are *weight bytes only*, not source decoder bytes or ES population memory. All model paths/results in Git exclude private account details and paid APIs.

## Scientific next gate

Before increasing Mirror state counts, test whether ES-learned code coordinates for center and width can be predicted by a **small meta-encoder from the same 32 support examples**, avoiding 960 black-box evaluations at inference. Match total stored bytes of encoder + scaffold + task codes and count encoder FLOPs. Independently, test sine Fourier features and code rank 3/4 for extrapolation, and test Mirror's representational bottleneck at a fixed parent with stronger alternatives than shear.
