# TinyStories scale series: actual task-aware embedding allocation

This note records high-value **actual-model** experiments run locally in the container on the user-provided TinyStories GPT-Neo checkpoints `1M`, `3M`, `8M`, and `28M`. It is not a 64× quality certificate.

## Setup

- Models: four GPT-Neo TinyStories checkpoints with hidden sizes 64/128/256/512.
- Tied token embedding / LM-head source matrix.
- Component budget: 0.25 bit per embedding scalar.
- Shared root: task-signature clustering from fixed calibration hidden states; root vectors quantized to 4 bits.
- Private exceptions: token residuals with token id + per-row scale included in the residual-bit budget.
- Task witnesses: KL to the uncompressed baseline, top-1 flips, and next-token NLL delta on held-out passages.

## 1. Input-gradient / frequency auxiliary selectors did not generalize

On the 8M model, output-Fisher-only exception selection gave mean KL `0.9118` on six held-out passages. Strongly mixing input-gradient or token-frequency scores degraded badly. A weak auxiliary coefficient looked helpful when tuned on all holdouts, but nested validation failed:

- output-only test KL: `0.8882`
- dev-selected auxiliary coefficient test KL: `0.9337`

Leave-one-holdout-out tuning also underperformed output-only (`0.9267` vs `0.9118`). Therefore input/frequency weighting is not treated as a robust selector improvement.

## 2. Discrete task-aware exception precision can help, but the surrogate matters

Uniform 4-bit residual exceptions were compared with a multiple-choice allocation over token residual precision. The allocation preserves the same total residual-bit budget.

A Fisher-weighted allocation improved 3M/8M but failed on 1M/28M. Restricting the precision options to 3–6 bits reduced over-concentration but did not make Fisher universal.

A better surrogate for models 3M and larger was **calibration predictive probability mass**: tokens the model itself assigns meaningful output probability receive higher task price even when they are not calibration targets.

Predictive-mass bounded-MCKP versus Fisher-selected uniform 4-bit exceptions:

| model | original six holdouts | completely new six holdouts |
| --- | ---: | ---: |
| 1M | `1.2481 -> 1.3077` (FAIL) | `1.2818 -> 1.3816` (FAIL) |
| 3M | `0.7865 -> 0.6848` (**-12.9%**) | `1.1428 -> 1.1005` (**-3.7%**) |
| 8M | `0.9118 -> 0.8176` (**-10.3%**) | `1.1908 -> 1.0712` (**-10.0%**) |
| 28M | `0.8457 -> 0.8006` (**-5.3%**) | `1.1660 -> 1.0494` (**-10.0%**) |

For 8M/28M the same allocation also reduced top-1 flips on the new holdout set. This is evidence that task-price construction is itself model/scale dependent; predictive mass is not claimed as a universal oracle.

## 3. The 1M failure is partly a root/private allocation problem

At 1M, a K256 root consumes too much of the 0.25-bps component budget. Reducing root cardinality frees private exceptions and changes the task frontier substantially.

On the new holdout, predictive-selected uniform 4-bit exceptions gave:

| root K | exceptions | mean KL |
| ---: | ---: | ---: |
| 16 | 1965 | 1.3022 |
| 32 | 1775 | 1.2042 |
| **64** | **1570** | **1.1348** |
| 128 | 1335 | 1.3168 |
| 256 | 1040 | 1.2939 |

K64 is best by KL, but not by top-1 flip, so root cardinality is a real multi-objective discrete variable. Precision MCKP still did not beat uniform residual precision at K64.

## 4. Scientific interpretation

These experiments support three design rules for QCO/FQC:

1. **Task-price construction and precision allocation must be separated.** A good representation with a poor surrogate can allocate bits incorrectly.
2. **Shared-root size and private-exception budget must be optimized jointly.** The smallest model preferred a much smaller root than the larger-model experiments.
3. **Weight reconstruction quality is not the commit metric.** Several task-aware allocations improved KL/decision metrics while worsening weight NMSE, and the reverse also occurred.

## Boundaries

Do not infer that:

- 0.25-bps embeddings preserve acceptable end-task quality;
- predictive probability mass is universally superior to Fisher information;
- these checkpoints demonstrate whole-model 64× compression;
- summed or local task costs are valid for simultaneous interventions without coupling correction.

Exact serializer accounting and exact task replay remain authoritative for final decisions.
