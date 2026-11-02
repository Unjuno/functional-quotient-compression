# MA-518 — factorized function × domain vector code

Status: FAIL on task utility; factorization claim NOT ESTABLISHED because the implementation did not decode the serialized factor codes.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact task coordinate `m=(procedure, domain)` to a frozen LM residual intervention interface so the same shared function/domain atoms can express multiple held-out procedure×domain functions without storing one full vector per combination.

- Native method: direct in-context demonstrations with a frozen model; explicit one-vector-per-task residual intervention is the storage upper reference.
- Insertion: residual stream after transformer block 6. A task behavior is a prompt-minus-query delta at this fixed site.
- `m`: flat task code vs factorized function×domain coefficients over the same development-only shared basis.
- Cheapest control: general low-rank/PCA coefficient table with the same basis rank and FP16 coefficient precision.
- This is a low-cost feasibility screen, not a natural-language or causal-head Function Vector claim.

## Hypothesis

H: On a synthetic procedure×domain lookup task, a factorized function×domain Mirror code will preserve held-out combination target NLL within 5% of explicit deltas, with at least 20% lower serialized payload than the byte-matched generic low-rank coefficient control at equal rank, after paying every per-combination product/task coefficient.

## Prior art delta

PA99 motivates functional activation vectors but does not establish that prompt-minus-query residual states are reusable causal operators. PA100 motivates activation differences as steering vectors. MA-516/517 failures make direct-task validity a hard interpretability gate here.

## Comparisons

- query only and direct in-context demonstration (validity gate)
- explicit intervention delta per training task
- generic pooled PCA basis + flat task coefficients
- factorized procedure/domain basis with explicitly charged per-combination product coefficients (Mirror)
- function-only and domain-only ablations

## Frozen protocol summary

The frozen checkpoint is `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32. Prompt families contain two independent factors: operation (`prepend`, `append`, `reverse`, `rotate`) and domain alphabet (`fruit`, `shape`, `color`, `vehicle`). The task maps one domain token through a procedure-defined transformation to another token; factor combinations are held out. Development worlds 51800–51801 may validate tokenization, task semantics and direct-ICL validity only; if either is at chance, fresh remains unopened and result is NOT ESTABLISHED pending a protocol amendment. Fresh worlds 51810–51812 × seeds 0–2 remain sealed until source freeze. No rank, site, prompt or gate is selected using fresh results.

Success requires direct ICL to exceed chance on each fresh world; factorized code to be within 5% of explicit-intervention NLL; and at least 20% fewer actual serialized payload bytes than generic PCA at the same rank. Failure includes invalid direct-ICL task behavior or no factorization advantage at matched quality/bytes.

## Storage and compute

Actual serialized payload includes all shared means/bases, coefficients, factor labels, task metadata and reconstruction state. The pinned model/tokenizer bytes are reported separately and added for deployment totals. Report extraction and intervention forward wall-clock; no optimizer updates are used.

## Boundaries

Synthetic deterministic transformations only. A failure here is scoped to this prompt/vector construction; it does not falsify head-specific causal FVs or natural language behavior steering.
