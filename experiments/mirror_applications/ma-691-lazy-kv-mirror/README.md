# MA-691 — lazy canonical KV Mirror read

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `7589ce171d6aa6743b7b0d43cd424134afb8fbdc`

## Hypothesis

If a logical attention View changes canonical cached keys and values by known right-transforms,

[
K_m=KA_m,qquad V_m=VB_m,
]

then native View attention can be evaluated without materializing a View-specific cache:

[
operatorname{softmax}(qK_m^T)V_m
=
operatorname{softmax}((qA_m^T)K^T)V B_m.
]

One canonical physical K/V cache should therefore serve many logical Views; only the current query and attention output need small View transforms.

## Prior-art delta

- PA51 LazyAttention defers positional transforms so one physical cache can serve multiple logical positions.
- PA52 cross-model KV transfer fits approximate affine cache translators.
- PA56 MLA caches a compressed latent and absorbs K/V expansion into read-side computation.

Exact delta tested here: **defer the Mirror/View transform itself**. The source and logical View are architected to have a known exact cache relation rather than learning an approximate translator.

## Comparisons

1. materialized View: explicitly transform the full prefix K/V cache before attention;
2. pre-materialized View: pay conversion once and reuse transformed K/V;
3. lazy View: keep canonical K/V, transform current query and attention output only;
4. MLA algebra diagnostic: explicit expanded K/V versus latent-cache absorbed computation.

## Gates

PASS requires all three fresh seeds to satisfy:
- lazy/materialized output max absolute error <= 2e-6 for every prefix length;
- RoPE-plane commutation max error <= 2e-6;
- MLA explicit/absorbed max error <= 2e-6;
- one physical canonical cache independent of logical View count.

Runtime is exploratory. A slower eager implementation does not falsify algebraic cache compatibility, but must be preserved as a negative runtime result.

## Key boundary

This does **not** imply any arbitrary Mirror-MoE/FFN branch can reuse another branch's cache. An earlier expert that changes hidden states without a known orbit relation changes later-layer K/V. Exact reuse requires a known cache transform, a canonical latent/cache interface, or cache-safe placement.

See `../../../docs/phase2/MIRROR_KV_CACHE_REUSE.md`.
