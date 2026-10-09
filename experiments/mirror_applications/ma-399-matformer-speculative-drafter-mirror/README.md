# MA-399 — MatFormer nested speculative drafter with Mirror view

Status: protocol frozen before development. Dedicated branch: `research/ma-399-matformer-speculative-drafter-mirror-20261009`.

## Mirror insertion

**Mirror insertion:** four Givens angles rotate the narrow width-8 draft activations before projection, while the width-16 verifier remains unchanged.

## H — Hypothesis

A tiny role-specific Mirror code improves the acceptance overlap of a nested MatFormer drafter with the full verifier enough to raise exact speculative throughput, using fewer actual bytes than a separate drafter/verifier pair.

## T — Frozen design

PA53 establishes nested MatFormer subnetworks; PA15 motivates routed slim-verifier comparisons. A frozen width-16 MLP supplies width-8 and width-12 nested draft roles. Compare hard width-8, width-12 reference, width-8 plus Mirror, width-8 plus FiLM, and an independent width-8 model. Measure held-out draft/full overlap, exact residual-corrected output, actual bytes and draft-plus-verifier latency. The PA15 hierarchical rejection router itself is outside this registered screen. Full details and gates are in `PROTOCOL.json`.

This is a one-step synthetic next-token distribution benchmark, not language-model generation evidence.

## D — Result

**Fact.** Seeds 39901/39902: Mirror acceptance overlap was .8441/.8432, versus nested width-8 .8413/.8416, FiLM .8473/.8482, width-12 .8746/.9058, and independent width-8 .9628/.9620. Full-verifier residual correction stayed exact for every method (Mirror max distribution error 1.19e-7/1.49e-7). Mirror expected speculative throughput was 0.642/0.652× full-only, versus 1.042/1.056× for nested width-8. Mirror payload was 3,413/3,407B versus 4,883/4,871B for the independent pair, but larger than shared nested-only payload (3,207/3,201B). Mirror adaptation took 1.61/1.57s versus FiLM .37/.38s. All twelve payloads replayed bytes, hashes and metrics; four tests pass. Fresh 39911–39913 remained sealed.

**Interpretation.** Residual-corrected speculation preserved exactness, but Mirror barely changed the draft/full overlap, lost to FiLM and width-12, and added enough trig/dispatch latency to make the measured speculative path slower than full-only. It reduced bytes against a separately stored drafter/verifier pair, but added bytes over the nested supermodel without a useful acceptance gain. The exactness property comes from the standard residual correction, not from Mirror.

**Hypothesis.** Four un-fused trigonometric rotations are poorly matched to this tiny CPU draft kernel; even a compiled implementation would still need a much larger overlap gain because the measured development overlap delta was under .003. The high independent drafter overlap indicates a quality frontier that the shared nested child did not reach under the fixed training budget.

## H / T / D / C / U

- **H:** Four Mirror angles on a nested width-8 MatFormer draft improve verifier overlap enough to raise exact speculative throughput while retaining shared-supermodel storage savings.
- **T:** PA53/PA15; frozen width-16 synthetic verifier, width-8/12 nested roles, Mirror/FiLM/independent controls, seeds 39901/39902, 1,200 update adaptation on training contexts, 2,048 held-out contexts, measured CPU latency and complete NPZ payloads.
- **D:** **FAIL.** Exact correction replay passed, but Mirror overlap improved width-8 by only .0016–.0028, fell below FiLM and width-12, and expected throughput fell to .64–.65× full-only. Fresh stayed sealed.
- **C:** FiLM achieves slightly higher overlap with simpler code; independent width-8 reaches ~.962 overlap and is faster, at larger payload.
- **U:** Fused/GPU kernels, multi-token autoregressive generation, actual MatFormer/LLM checkpoints and PA15 hierarchical routing.
