# SmolLM2 64× component-budget constraints

This note records exact arithmetic consequences of the certified T001 denominator. It is not a codec result and assumes a 16-bit scalar baseline.

## Certified denominator

- Unique paid scalars: `N = 134,515,008`
- 16-bit baseline: `2,152,240,128` bits
- Hard 64× target: `33,628,752` bits = `4,203,594` bytes = `0.25` bits/original scalar

Component scalar counts:

- tied embedding / LM head: `28,311,552` (`21.047%`)
- attention: `26,542,080` (`19.732%`)
- MLP: `79,626,240` (`59.195%`)
- other/norm state: `35,136` (`0.0261%`)

## 1. Leaving the embedding at FP16 makes 64× impossible

Even under the unrealistically favorable assumption that **all attention and all MLP payload cost zero bits**, the FP16 embedding plus the small `other` component require `453,547,008` bits, or `3.37172` bits/original scalar. That is only `4.745×` compression versus the 16-bit baseline.

Therefore any 64× codec for this model must compress the embedding/LM-head representation itself. This conclusion is independent of attention/MLP metadata or optimizer quality.

## 2. All major components must be sub-bit on average

If `other` remains at 16 bits/scalar, it consumes `0.00417928` bpp. The remaining embedding + attention + MLP components must therefore average at most:

`0.24588495 bits/scalar`

which corresponds to `65.071×` compression relative to 16-bit storage. Uniform `0.25` bit/scalar on the three major components is already slightly above the hard 64× model budget once the fixed `other` payload is paid.

Selected budget examples:

- embedding at `1.0` bit/scalar leaves only `0.05007` bit/scalar on average for the non-embedding part;
- embedding at `0.5` bit/scalar leaves `0.18336` bit/scalar for the non-embedding part;
- embedding at `0.25` bit/scalar leaves approximately `0.25` bit/scalar for the non-embedding part;
- attention and MLP both at `0.5` bit/scalar leave no non-negative budget for the embedding under the same accounting.

## 3. Restricted structural-deletion families are far from 64× by themselves

The following are **nominal scalar-count proxies**, not emitted codec bits and not claims that the interventions are safe across all layers.

If the tested structural ideas were idealized across all corresponding layers:

- rank-3-like 25% scalar saving across all attention covers about `5.01%` of the bit saving required for 64×;
- rank-1-like 75% scalar saving across all attention covers about `15.03%`;
- dropping 50% of all MLP channel scalars covers about `30.07%`;
- combining the rank-1 attention proxy with 50% MLP dropping covers about `45.10%` of the required saving.

Even deleting **all attention and all MLP payload entirely** covers only about `80.18%` of the required bit saving because the embedding remains too large at FP16.

## Interpretation

For SmolLM2-135M, 64× cannot be reached by treating FQC primarily as structured pruning or low-rank deletion while leaving surviving values at ordinary multi-bit precision. A viable path must include low-description coding of the embedding and of the surviving attention/MLP state, with actual serializer accounting.

These are arithmetic constraints, not a lower bound against all possible codecs and not evidence that 64× is achievable.
