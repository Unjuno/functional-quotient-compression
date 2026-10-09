# MA-517 — Mirror composition of function vectors

Status: SCREENING.

## H / T / D / C / U

**H:** Two separately extracted task vectors for fruit→color and color→shape may compose better with role-specific layer placement than raw same-layer vector addition.

**T:** Pinned frozen GPT-2; synthetic mapping tasks with four support pairs for each function and four held-out fruit→shape queries. Compare query-only, direct combined ICL, raw `f1+f2` at layer 6, layer-factorized `f1@4/f2@8`, and each vector ablation. Fresh worlds 51710-51712 × seeds 0-2. Both vectors and all metadata are charged.

**D:** Pending.

**C:** MA-516 found that the same support-delta extraction method did not improve single-function accuracy. Composition may fail because its component vectors are not causal task representations.

**U:** Fresh accuracy, NLL, cross-task interference, payload bytes, and runtime.
