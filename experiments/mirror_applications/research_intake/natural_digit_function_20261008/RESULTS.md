# Real-image shifted-LoRA Mirror View pilot — results (2026-10-08)

**Decision: exploratory screen did NOT pass the frozen Mirror-specific Pareto gate.** No official MA status change. This report is an intake under `experiments/mirror_applications/research_intake/natural_digit_function_20261008`, complementing MA-1096/1099/1114/1115; it is not their completed experiment.

## Design and provenance

Frozen protocol: [PROTOCOL.md](PROTOCOL.md), recorded before execution at commit `d12d62a1902c702e57b09cfaba49d0862d485226`. Source: [run_pilot.py](source/run_pilot.py), SHA-256 `a2c847b4fa3a95c443487d3551728a0107936ac4f66983a8ac1d48e04bca6f55`. [Raw 48-row results](RESULTS_CORE.csv), first-run SHA-256 `db044815d36d8e08e53817744aa8f7b6406ad03bdee020196951f1aab7a84c9a`. [Verification](VERIFICATION.json).

Data: `sklearn.datasets.load_digits` real 8×8 handwritten digits, 60%/20%/20% stratified train/dev/test for seeds 41, 42. Train one clean 64→64→64→10 base model (clean test top-1 **98.61% / 95.56%**), then freeze. Learn independent rank-4 LoRA `64×64` hidden-layer deltas for five shifted/blurred training conditions. Fit shared rank-6 input and output delta subspaces from those source LoRAs **without** opening target task deltas. On four withheld corruptions, train the code or native LoRA with identical minibatches/220 updates.

Source tasks: rotate +20°, rotate −20°, horizontal shift +1, contrast ×0.7, Gaussian blur σ0.8. Held-out target conditions: rotate +33°, rotate −33°, vertical shift +1, blur σ1.2. These are *real images with synthetic shifts*, not independently sourced natural text or end-to-end LLM tasks.

## Aggregate held-out score and marginal serialized bytes

| Candidate | Test top-1 (mean of 8) | CE loss ↓ | Four-task adaptation-bank NPZ | CPU forward ms / task ↓ | Weight error vs independent LoRA ↓ | Logit RMSE vs LoRA ↓ |
|---|---:|---:|---:|---:|---:|---:|
| base | 37.01% | 7.662 | 22 B | 0.114 | 1.000 | 8.996 |
| scalar | 40.10% | 6.491 | 2,026 B | 0.126 | 1.128 | 7.842 |
| diagonal | 49.79% | 3.687 | 4,662 B | 0.124 | 1.305 | 6.013 |
| mirror | 52.19% | 2.983 | 4,694 B | 0.197 | 1.302 | 5.483 |
| dense | 69.97% | 1.195 | 5,142 B | 0.125 | 1.341 | 3.719 |
| lora4 | 86.98% | 0.416 | 10,222 B | 0.117 | 0.000 | 0.000 |

Each method inherits the **same frozen pretrained base**, which is excluded equally; NPZ includes all saved shared basis tensor(s), four task codes/factors, tensor shape/type and ZIP record metadata. The common source-adapter training expense is **not** accounted in the bank inference byte tally. The base-only 22 B reflects an empty named NPZ archive, **not** a claim that the classifier occupies 22 B.

Mirror uses 6 diagonal scalars plus 2 fixed-plane Givens angles/task; BOLT-like diagonal uses 6 scalars/task. Mirror bank state is **32 B larger** than diagonal at these four tasks (4694 vs 4662 B). The dense shared core uses 36 scalars/task (5142 B) and independently fitted LoRA rank 4 uses 512 factors/task (10222 B).

## Seed- and condition-specific top-1 test accuracy

| Seed | Held-out corruption | Diagonal shared | Structured Mirror | Dense shared core | Independent LoRA r4 |
|---|---|---:|---:|---:|---:|
| 41 | `rot_p33` | 48.61% | 58.33% | 76.39% | 90.83% |
| 41 | `rot_m33` | 33.61% | 36.39% | 71.67% | 80.83% |
| 41 | `shift_yp1` | 47.22% | 48.89% | 56.67% | 90.28% |
| 41 | `blur_12` | 81.11% | 81.67% | 87.50% | 92.78% |
| 42 | `rot_p33` | 52.22% | 51.67% | 72.22% | 80.83% |
| 42 | `rot_m33` | 34.17% | 29.72% | 56.39% | 85.00% |
| 42 | `shift_yp1` | 33.61% | 41.67% | 55.28% | 86.94% |
| 42 | `blur_12` | 67.78% | 69.17% | 83.61% | 88.33% |

## Frozen scientific gate

- Mirror improves test **accuracy** over BOLT-like diagonal for 4/4 tasks under seed 41 but only 2/4 tasks under seed 42. Required ≥3/4 in **both** seeds — **FAIL**.
- Mirror lowers cross-entropy on 4/4 and 4/4 tasks. This is a **narrow mechanism signal**, not a Pareto win.
- Mean accuracy improves by **2.40 percentage points** over diagonal, but dense-core accuracy is 69.97% and native rank-4 LoRA is 86.98%. The usefulness gap is large — **FAIL** to beat the stronger accuracy/storage frontier.
- Mirror average eager CPU forward latency is 0.197 ms versus diagonal 0.124 ms (**1.59× slower**). Timing numbers are noisy small-CPU measurements, but the predeclared no-slower condition was not met.
- The dense shared core has *higher mean Frobenius weight-delta reconstruction error* (1.341 vs Mirror 1.302) yet *much better test accuracy* (69.97% vs 52.19%). This specifically illustrates why weight-delta reconstruction alone is **not an adequate function-utility metric**. The dense-core logit RMSE (3.719) is lower than Mirror (5.483).
- Stable deterministic fields were exactly replayed in a full second run: **48 rows × 14 fields**, maximum numerical difference **0**. CPU timings were excluded from exact replay. The method NPZ archives were tested for exact numeric round-trip.

## Interpretation and limits

**Supported:** introducing two structured Givens degrees of freedom beyond a diagonal code can lower held-out CE for related learned shift tasks with +32 B of total bank storage. It can also fail on top-1 accuracy for a different initial seed. The full dense core or new LoRA factors can preserve far more useful task performance at additional storage cost.

**Not supported:** an unconditional Mirror-specific advantage, an optimal `m` family, natural LLM LoRA compression, information-theoretically free capacity, real GPU throughput or end-to-end deployment savings. The task perturbations are hand-constructed; shared bases are oracle-free for target tasks but source-LoRA training is an offline expense; the BOLT-inspired diagonal core and CtS-inspired dense core are *simplified structural controls*, not exact paper reimplementations. This pilot only measures one rank, one Givens plane pattern and two seeds. A new orbit/private residual experiment must preregister different seeds before any modification.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/mirror_applications/research_intake/natural_digit_function_20261008/source/run_pilot.py \
  --output /tmp/real_digit_function_mirror_audit --seeds 41,42
```

Requires `torch`, `numpy`, `scipy`, `scikit-learn`. All data comes from the packaged `sklearn.datasets.load_digits`, not a private download. The live MA-255 queue and 29 PROMISING / 18 FAIL program statuses are unchanged.
