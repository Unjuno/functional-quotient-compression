# A10 / A10b — learned scaffold + unseen task adaptation (2026-11-15)

**Status:** completed exploratory CPU synthetic study. **Negative result for Mirror-specific storage advantage.** A10's 144 final models were locked before 432 evaluation records. A10b is a post-A10 outcome-driven exploratory compute-horizon follow-up with 48 final models and 144 additional evaluation records. The teacher uses no Mirror mechanism.

## Question
A9's family-aware fixed scaffold was successful, but functionally equivalent to a direct shift. Does a shared scaffold *learned from 16 labeled tasks* allow Mirror to adapt eight unseen ramp tasks using only 32 observations per task, while staying at the same serialized FP32 byte cap as strong controls?

## Results (world-mean NRMSE, lower is better)

| Method | Cap | Seen-task dev | Unseen IID | Unseen OOD | Quality pass worlds |
|---|---:|---:|---:|---:|---:|
| Mirror | 512 B | 0.158 | 0.269 | 0.294 | 0/6 |
| FiLM | 512 B | 0.071 | 0.116 | 0.140 | 5/6 |
| Mirror | 1024 B | 0.114 | 0.212 | 0.226 | 2/6 |
| FiLM | 1024 B | 0.027 | 0.052 | 0.076 | 6/6 |
| Shift | 1024 B | 0.016 | 0.069 | 0.081 | 6/6 |

Full model comparisons include Mirror inverse, Mirror without inverse, FiLM, direct Shift, shared output Basis, and per-task Multihead. Independent post-lock replay regenerated 8/8 A10 and 6/6 A10b-long selected model files byte-for-byte. A10b 3x optimization work did not rescue Mirror: short 2/4 passes, long 1/4 passes; FiLM and Shift 4/4 in both regimes.

## Evidence and important caveat

The originally specified "10% less stored bytes at both same-byte-cap levels" gate **cannot identify a 10% reduction** from this two-point near-byte-matched design (e.g. Mirror 500 B versus FiLM 512 B). Therefore the general minimal-byte-rate advantage is **unresolved**, although observed error and quality-pass results are negative for Mirror. The models have unequal FLOPs, data are one synthetic ramp family, adaptation is few-shot supervised and task IDs are provided.

- [`REPORT_JA.md`](REPORT_JA.md): full Japanese report, equations, variable table and uncertainty.
- [`PROTOCOL.md`](PROTOCOL.md): frozen A10 protocol; [`A10B_PROTOCOL.md`](A10B_PROTOCOL.md): post-A10 exploratory follow-up design.
- [`a10.py`](a10.py): world generator, six learners, exact serialization, two-stage training.
- [`train.py`](train.py): writes checkpoint files and cryptographic training lock, **no test access**.
- [`audit.py`](audit.py): independent post-lock load/hash/serializer verification and train-task development/unseen IID/OOD evaluation.
- [`followup.py`](followup.py): A10b experiment schedule.
- [`replay.py`](replay.py): selected independent-process training replays; [`analyze.py`](analyze.py): cross-world descriptive uncertainty.
- [`test_a10.py`](test_a10.py): model, leakage, reproducibility and corruption checks.
- `results/{main,a10b_short,a10b_long}`: aggregate/world/sample scores and training locks. Raw 192 FP32 weight files are **not committed to Git**, but available in the conversation evidence ZIP.

## Reproduce on CPU

Run from this directory, after installing the dependencies in `requirements.txt`:

```bash
python -m pytest -q test_a10.py
python train.py --out /tmp/a10-repro/main
python audit.py --run /tmp/a10-repro/main
python followup.py --out /tmp/a10-repro/a10b_short --length short
python audit.py --run /tmp/a10-repro/a10b_short
python followup.py --out /tmp/a10-repro/a10b_long --length long
python audit.py --run /tmp/a10-repro/a10b_long
python replay.py --run /tmp/a10-repro/main --count 8
python replay.py --run /tmp/a10-repro/a10b_long --count 6
python analyze.py --root /tmp/a10-repro
```

For `analyze.py`, use a directory with subfolders named `main`, `a10b_short`, and `a10b_long`. Create `/tmp/a10-repro` as an empty parent directory first (`mkdir -p /tmp/a10-repro`). All runs refuse to overwrite existing experiment subdirectories.

## Computational context

AMD EPYC 9V74 80-Core Processor, observed CPU clock 2596.134 MHz (not locked); Python 3.13.5; PyTorch 2.10.0+cpu; 4 independent processes × 1 PyTorch thread, FP32, 128 samples per optimizer step. A10 meta 1,000 updates and new-task adaptation 300; A10b long 3,000 and 900. Not a GPU performance comparison, a matched-FLOPs audit, a natural-language model, or a general-capacity claim.
