# MA-399 — MatFormer speculative drafter via Mirror granularity code

Status: **FAIL at development gate**. Fresh worlds 39911–39913 remain sealed.

## H / hypothesis

H: One per-context angle over a shared rank-2 residual basis should improve width-16 nested-draft acceptance by at least 0.05 tokens per four-token block, reach within 0.05 of routed width-32, preserve exact target sampling, and fit within 1.05x the shared supernetwork payload.

## T / method

Protocol was frozen before development in `PROTOCOL.json` (base `2d686f0`). Two synthetic stochastic next-token worlds (seeds 39901 and 39902) used one 64-hidden-unit MLP with nested widths 16/32/64 and 1,600 Adam updates. Width 64 was the verifier. Controls included nested16, routed32, per-context scalar residual, ordinary rank-2 coefficients, independent width16 draft and Mirror rank-2 angular view. The exact rejection correction is tested by comparing its resulting distribution to the verifier distribution. Payloads are FP16 NPY arrays in ZIP archives, including metadata and all codes/bases; actual archive bytes are reported.

## Results

- Mirror expected accepted proposals: 3.941 and 3.932 of 4; nested16: 3.973 and 3.975; routed32: 3.983 and 3.983.
- Mirror did not meet the required +0.05 over nested16 and was below routed32.
- Mirror archive: 12,663 B; shared supernetwork: 11,713 B (1.081x, above 1.05x). Ordinary rank-2: 12,555 B; scalar: 12,383 B.
- Exact rejection correction maximum TV from verifier: 0 for every method and both worlds (floating point arithmetic in this harness).
- Fresh was not run after the registered development failure. End-to-end generation latency was not measured because the experiment failed the preregistered acceptance and storage gates.
- Independent width16 draft acceptance was 1.715 and 1.736; its payload was 23,129 B including the shared supernet and independent draft.

## D / decision

**FAIL** for the registered Mirror speculative-decoding hypothesis. Mirror adds bytes and reduces acceptance versus both nested16 and routed32 in these synthetic worlds.

## C / strongest counter-hypothesis

The nested width16 model was already close to the width64 verifier, leaving little headroom (less than 0.03 expected accepts below the block maximum). The Mirror angular code approximated the residual poorly and added archive overhead. The +0.05 target was therefore not achievable in these development worlds, and ordinary alternatives were also worse than plain nesting.

## U / limits

This is a small synthetic MLP proxy, not a MatFormer Transformer or language-model result. No fresh worlds or end-to-end latency comparison were run because the frozen development gate failed. The harness establishes exact rejection correction algebra, not a production speculative-decoding speedup.

## Fact / interpretation / hypothesis

- **Fact:** Both development seeds fail the acceptance and byte gates; serialized payloads and hashes were checked; three tests pass.
- **Interpretation:** On this setup, plain nested width16 and routed width32 dominate the added Mirror residual view.
- **Hypothesis:** A more difficult target with larger verifier/drafter mismatch may leave useful residual structure for a Mirror code, but this was not tested.
