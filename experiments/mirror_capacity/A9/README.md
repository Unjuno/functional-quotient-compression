# A9 and A9R — conditional-code reachability under function-aware initialization

**2026-11-15 — CPU synthetic experiments; no Mirror-specific general compression claim.**

A8 established that width-4, 244-byte reversible-shear Mirror exactly represents 16 independent ramp functions but standard random-initialized training often did not reach good reconstructions. A9 tests optimization accessibility from **world-independent, family-aware** scaffold initialization versus random initialization. A9R repeats the locked conditions on six fresh worlds after the A9 result.

## Main result (both tests have 6 worlds × 2 training seeds)

| Method | A9 pass worlds | A9R pass worlds |
|---|---:|---:|
| Mirror4, random full-weight training | 0/6 | 0/6 |
| Mirror4, scaffold full-weight training | 0/6 | 1/6 |
| **Mirror4, fixed scaffold, train 32 task-code scalars** | **5/6** | **6/6** |
| Direct shift4, scaffold full-weight training | 6/6 | 6/6 |

Pass means mean-seed IID and OOD NRMSE <= 0.08. Note the direct shift and pre-Mirror scaffold are **functionally identical** to the full Mirror scaffold when its companion ReLUs are frozen inactive. The witness already uses the teacher family's formula, although not the world's thresholds. Therefore this is an **optimization/parameterization** finding, not extra Mirror capacity.

- [Detailed Japanese report, proof, variables, uncertainty](REPORT_JA.md)
- [A9 pre-run protocol](PROTOCOL.md)
- [A9R frozen fresh-world follow-up protocol](A9R_PROTOCOL.md)
- [Main model-world table](results/main/world_means.csv) and [main summary](results/main/summary.json)
- [Follow-up model-world table](results/replication/world_means.csv) and [follow-up summary](results/replication/summary.json)
- [Evidence hashes](EVIDENCE.json)

## Reproduction

From repository root with `torch`, `numpy`, `scipy`, `pytest` installed (original: Python3.13.5, torch2.10.0+cpu, one CPU numerical thread per process):

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q experiments/mirror_capacity/A9/test_a9.py
python -m experiments.mirror_capacity.A9.train --out /tmp/a9-fresh --jobs 4
python -m experiments.mirror_capacity.A9.audit --out /tmp/a9-fresh
python -m experiments.mirror_capacity.A9.replay --out /tmp/a9-fresh
python -m experiments.mirror_capacity.A9.train --out /tmp/a9r-fresh --worlds 826 827 828 829 830 831 --regimes random_full4 scaffold_code4 scaffold_full4
python -m experiments.mirror_capacity.A9.audit --out /tmp/a9r-fresh
```

Outputs must not already exist. `train.py` writes all model SHA-256 receipts **before** `audit.py` reads any dev/IID/OOD metrics; `audit.py` reserializes all model files for exact identity. Both streams are deterministic on the reported CPU runtime but cross-platform bitwise equality is not guaranteed.

Stored sizes count *all* FP32 parameter tensors, fixed 32-byte model header and 32-byte SHA-256. All weights and per-model score detail are in the external evidence bundle; only aggregate tables and hashes are stored in Git. Shared source/decoder and compute cost are outside the weight-file budget. The A8 source and codec are imported by A9 rather than copied.
