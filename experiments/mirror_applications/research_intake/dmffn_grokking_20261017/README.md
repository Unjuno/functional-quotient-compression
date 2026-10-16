# DMFFN-GROK-001 — Grokking and dual-input Mirror SwiGLU

**Executed synthetic algorithmic CPU experiment, 2026-10-17. Delayed generalization observed; strong Grokking NOT established.**

Modulus 31, 961 ordered (a,b) inputs, four targets (a+b, a+2b, 2a+b, 2a+2b) mod 31. Train/test inputs split 60%/40%. Different splits of the same rules are not independent natural datasets and do not test unseen task combinations.

3 new splits × 2 initializations × 5 methods = **30 main fits**; 8,000 full-batch AdamW updates, LR 0.005, weight decay 0.5 except no-decay ablation. A separate six-fit shared-only control is postprimary.

| Method | Median heldout accuracy | Delayed generalization / 6 | Reach 90% / 6 | Inference bytes |
|---|---:|---:|---:|---:|
| Factorized Givens Mirror | 59.99% | 5 | 0 | 43,597 |
| Native gain, matched parameters | 35.19% | 0 | 0 | 43,597 |
| Pair-specific Mirror, matched parameters | 35.71% | 0 | 0 | 43,597 |
| Four independent SwiGLU | 82.32% | 6 | 0 | 163,861 |
| Mirror with weight decay 0 | 9.18% | 0 | 0 | 43,597 |

Operational delayed-generalization criterion: >=99% training accuracy first, <=30% test at that time, then >=50% test after >=1,000 further updates. Strong Grokking additionally requires >=90% test; **0/30 main trials** reached that level. The observed generalization mostly improves gradually; no discontinuous transition or Mirror-specific causal effect is demonstrated.

Seven tests passed, all thirty saved inference models were SHA checked and independently replayed using NumPy FP64, six postprimary shared-only models separately replayed; two full 8,000-update retrainings reproduced tensor SHA and 41 logged checkpoints. Same-session verification, not third-party replication.

Full frozen protocols, source, raw results, model files and Japanese report are preserved in the downloadable conversation full-evidence ZIP. This compact GitHub note is not a runnable reproduction by itself. Historical statuses and main are untouched.
