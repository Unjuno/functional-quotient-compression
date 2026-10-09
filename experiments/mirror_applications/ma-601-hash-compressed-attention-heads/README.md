# MA-601 — Hash-compressed attention heads

Status: **FAIL (development; fresh sealed)**. Prior art PA120 HashedNets. Pre-metric amendment 1 corrected student initialization and accounting before any registered metrics.

## H — Hypothesis

A shared hashed Q/K/V projection with per-head Givens Views can recover useful logical attention heads, lowering actual payload while improving teacher KL over native tied/salted hashing and MQA.

## T — Execution

Distilled a fixed four-head attention teacher on 8-token sequences of 32-dimensional random features. Compared independent heads, GQA-2, MQA, tied/salted HashedNet projections, Givens Views, diagonal gates and rank-one residuals. Development worlds 60101 and 60102; each method got 1,500 AdamW updates. Actual NPZ payloads include the output readout and all projection tables/codes/metadata. Fresh worlds 60111–60113 stayed sealed.

## D — Decision

**FAIL.** Mirror has 5,006 B (48.0% of independent-head payload) and more diverse contexts than tied hash (mean cosine .340/.169 vs 1.0), but teacher CE is 3.236/3.282, about 1.10/1.16 nat above the independent-head upper control. It is worse than MQA (3.152/3.235), diagonal hash (3.191/3.208), and rank-one hash (3.141/3.135). It also exceeds the salted hash byte cap: 5,006 B vs 4,627 B (1.082x). Givens diversity does not recover the teacher's attention quality.

## C — Strongest counter-hypothesis

Four independent teacher heads contain different Q/K/V functions; shared hash buckets and input rotations cannot reproduce them. Native MQA/GQA also miss quality, while rank-one residual and diagonal controls are better at byte-near cost.

## U — Boundaries

Synthetic teacher distillation only, not language modeling or natural-token NLL. Fixed-update CPU screen, not near-convergence capacity or optimized kernel evidence.

## Facts / interpretation / hypothesis

- **Fact:** Teacher entropy/CE is 2.138/2.120 in the two worlds. Independent heads reproduce it at 10,421 B with top-1 agreement .997/1.000. Mirror CE 3.236/3.282; MQA 3.152/3.235; rank1 hash 3.141/3.135. Mirror context cosine .340/.169; tied hash 1.0/1.0. Fresh stayed sealed.
- **Interpretation:** The View creates distinct head activations, but functional diversity alone is not useful multiplicity: output quality misses badly, and the codes push payload above the salted-hash cap.
- **Hypothesis:** Per-head Q/K/V functions need richer private state than a 16-angle input rotation; low-rank residual or independent projections provide the missing directions.
