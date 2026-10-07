# SRM002 — Ordered non-commutative rules and shared/private decomposition

Date: 2026-10-07 JST

Evidence boundary: synthetic mechanism and causal-Transformer bridge only. No natural-language or LLM capacity claim.

## Questions

1. Can multiple shared rules compose in an order-sensitive, non-commutative setting?
2. If only part of the rule set is shareable, can a shared basis plus sparse private residual recover full quality at much lower storage?
3. Can private rules be discovered automatically after training rather than supplied by oracle?
4. In the discrete causal setting, is failure caused by insufficient representation or by credit assignment?

## A. Ordered factorized operator diagnostic

Teacher state is 16-dimensional. Each rule is drawn from an 8-dimensional shared signed operator basis and four rules are applied in order:

x_(t+1) = (I + rho * sum_j c_(r_t,j) B_j) x_t.

For two rule residuals A and B,

(I+A)(I+B) - (I+B)(I+A) = AB - BA.

Thus non-zero commutator implies order matters.

Primary 68-rule panel, fresh 4 worlds, 600 updates, actual serialized bytes:

| method | bytes | median held-out NMSE | median reverse-order NMSE |
|---|---:|---:|---:|
| shared signed basis | **12,373** | **5.94e-14** | **6.32e-14** |
| hashed one-expert-per-address MoE, E=10 | 12,885 | 2.44e-2 | 2.46e-2 |
| Mirror stretch+shear | **10,712** | 2.91e-2 | 3.04e-2 |
| full independent rule operators | 71,444 | 1.91e-11 | 1.90e-11 |

The hashed MoE remained at approximately 0.0245 held-out NMSE after 2,400 updates, so the 600-update gap is not only slower convergence in that control.

The shared basis remained effectively exact when rule count increased to 132 and 260, while serialized payload grew from 12,373 B to 14,421 B and 18,517 B respectively.

### Strong multi-expert mixture controls

At exactly 2,592 learned parameters:

- signed mixture of 8 shared full operators: approximately zero NMSE;
- positive softmax mixture of 8 shared full operators: 12,373 B and 0.00174 held-out NMSE at 1,200 updates;
- positive softmax mixture of 16 full operators: 22,749 B and approximately zero NMSE in two tested worlds.

Interpretation: the positive mechanism is **weighted composition of multiple reusable shared components**, not Mirror geometry itself. Mirror-specific advantage failed on this family.

## B. Shared + sparse private hybrid

132 rules were generated from the shared 8-dimensional basis. A fraction additionally received an independent rank-2 private residual.

### 25% private, fresh 3 worlds

| method | bytes | median held-out NMSE |
|---|---:|---:|
| shared only | 14,493 | 1.716e-3 |
| byte-near hashed MoE | 23,701 | 2.728e-2 |
| **shared + sparse private rank-2** | **24,654** | **2.30e-12** |
| all-independent full rules | 137,044 | 7.41e-7 |

The hybrid is about **5.56x smaller** than the full independent representation.

### 50% private, fresh 3 worlds

| method | bytes | median held-out NMSE |
|---|---:|---:|
| shared only | 14,493 | 3.350e-3 |
| byte-near hashed MoE | 32,917 | 2.715e-2 |
| **shared + sparse private rank-2** | **33,102** | **5.75e-10** |
| all-independent full rules | 137,044 | 7.94e-8 |

Even with half the rules private, the hybrid is about **4.14x smaller** than the full independent representation while recovering essentially full quality.

This is a mechanism result: the teacher is intentionally generated from a shared component plus sparse private residual. It proves the decomposition can work, not that natural language necessarily has this decomposition.

## C. Automatic private-rule discovery

Oracle knowledge of the private mask was removed.

All 132 rules first received rank-2 residual capacity. After 600 updates, each rule was scored by the Frobenius norm of its learned effective residual matrix.

With lambda=0.001, three worlds gave:

- private-mask AUC: **1.0 / 1.0 / 1.0**;
- minimum private score was more than 10x the maximum shared-rule score in every world.

The true private count was then also removed. A two-cluster fit on log residual norm selected:

- 33 / 33 / 33 rules;
- precision **1.0 / 1.0 / 1.0**;
- recall **1.0 / 1.0 / 1.0**.

Physical compact checkpoints:

- overcomplete: 49,039 B;
- automatically compacted: **24,943 B**;
- held-out NMSE: approximately 2.96e-6 to 3.93e-6.

Decision: learn-many -> score -> cluster -> prune -> compact passes on this controlled family.

## D. Discrete causal bridge

A harder bridge used independent invertible affine maps over GF(5)^2 (25 states). Four rules are applied in order. Reversing the four-rule order changes the target in about **96.4%** of sampled sequences.

### Final loss only

A high-capacity per-rule low-rank model with rank 24 remained near chance after 1,200 updates:

- held-out accuracy: 2.60%;
- reverse-pair both-correct: 0%.

A full per-rule FFN was also near chance in the early final-only runs.

### Intermediate-state diagnostic supervision

When the state after each rule application was supplied as an auxiliary target:

| parameterization | bytes | held-out accuracy | reverse both-correct |
|---|---:|---:|---:|
| per-rule low-rank r=8 | 75,441 | 14.1% | 2.1% |
| per-rule low-rank r=16 | 106,161 | 36.5% | 15.0% |
| per-rule low-rank r=24 | **136,945** | **100%** | **99.8%** |
| per-rule full FFN | 224,489 | 99.0% | 99.4% |
| shared basis r=8 | 47,161 | 4.2% | 0% |
| Mirror r=8 | 45,402 | 4.2% | 0% |

Interpretation:
- this independent discrete rule family is not economically represented by the tested shared/Mirror coordinates;
- private residual rank must approach hidden width to recover quality;
- even adequate representation fails under final-only supervision, so credit assignment is a separate bottleneck.

## Decision

### PASS
- ordered non-commutative composition from reusable shared basis;
- partial sharing plus sparse private residual;
- automatic private-rule discovery and physical pruning on the controlled hybrid family.

### FAIL
- Mirror stretch+shear as the main atom family for the SRM002 operator teacher;
- narrow shared basis / Mirror for independent discrete affine rules.

### NOT ESTABLISHED
- natural-language decomposition;
- LLM storage or quality advantage;
- automatic discovery without the controlled sparse-private teacher structure.

## Current architecture implication

The best current abstraction is:

> **canonical shared Transformer + sparse weighted shared-rule basis + only-when-needed private residual experts**

Mirror is now an optional rule parameterization and should remain only if it beats simpler signed/shared-basis controls.

## Next gate

SRM003 should remove explicit private masks and explicit rule-address supervision inside a small causal Transformer:

1. shared signed basis in the FFN residual;
2. initially overcomplete low-rank private residual bank;
3. LM-loss training;
4. residual-score pruning and physical compaction;
5. compare Dense, standard top-k MoE, LoRA-MoE, and shared-rule hybrid at actual serialized bytes and active compute.

Do not move to a natural-language claim until this gate passes.
