# Latest Worker Findings — MA-241 and MA-253

Date: 2026-10-07 JST
Purpose: compact, actionable context for subsequent MA workers. Dedicated experiment branches remain the source of truth.

## MA-241 — expert tying across depth + layer-specific Mirror views

Branch: `research/ma-241-expert-tying-mirror-20261007`  
Verified result commit: `0ee183668285231d825e853c69c4791b9d252bf2`  
Status: **PROMISING**

### Fact

Synthetic teacher: two layer-specific expert pools related by the same family of learned Givens coordinate transforms used by the Mirror candidate.

Fresh worlds 24101–24103:
- untied payload: 45,681 B;
- hard-tied payload: 23,967 B;
- Mirror payload: 24,286 B;
- Mirror therefore used 46.8% fewer serialized bytes than untied;
- Mirror passed all preregistered quality/storage gates in 3/3 worlds;
- at similar payload, Mirror beat gate and rank-1 residual controls in all fresh worlds.

Runtime:
- unfused CPU implementation was materially slower;
- median Mirror/tied inference throughput ratio was about 0.61x;
- training wall time was about 2.1–2.3x tied.

### Interpretation

A low-description layer coordinate can recover useful logical layer-specific expert behavior when the true layer variation lies on the same compact coordinate orbit.

This is a positive feasibility result for:
- expert tying + View differentiation;
- low-byte logical multiplicity;
- layer-coordinate structure.

It is **not** evidence that naturally trained experts differ by Givens rotations.

### Mandatory lesson for future workers

When testing a new Mirror family, include two teacher/data regimes when practical:

1. **aligned regime** — teacher variation is representable by the candidate coordinate;
2. **misaligned regime** — variation comes from an independently parameterized control family.

A useful method should not be evaluated only on its own aligned teacher.

Also report:
- fused/compiled runtime if runtime is central;
- analytical MACs separately from wall time.

## MA-253 — cache-safe final-layer Mirror-MoE

Branch: `research/ma-253-cache-safe-final-moe-20261007`  
Verified result commit: `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`  
Status: **FAIL for Mirror expert replacement; placement mechanics PASS**

### Fact

Teacher: independent per-domain rank-2 LoRA updates.

Fresh worlds:
- rank-2 LoRA: approximately 3.6e-14 to 6.1e-14 MSE;
- Mirror-only: approximately 0.0161 to 0.0173 MSE, essentially base-only quality;
- Mirror payload: 2,956 B;
- LoRA payload: 4,110 B;
- Mirror saved bytes but missed the quality gate by a very large margin;
- Mirror + rank-1 private residual improved strongly but remained far behind rank-2 LoRA.

Cache probe:
- applying the View only after final attention changed no cached K/V value;
- applying a View before a later attention layer changed downstream cached K/V;
- final-FFN-only conditional specialization is therefore cache-safe in this tiny causal probe.

### Interpretation

A small Givens coordinate cannot be assumed to encode arbitrary independent low-rank private updates. Private residual capacity remains necessary when variation is outside the View orbit.

The cache-placement result is orthogonal and useful:
- conditional computation after the final attention/at final FFN can preserve prefix KV-cache validity.

### Mandatory lesson for future workers

Do not ask a narrow Mirror coordinate to compete with a teacher explicitly generated from a richer independent family and then generalize the negative result to all Views.

Instead sweep:
- coordinate family;
- private residual rank;
- fraction of variation explained by shared orbit vs private update.

For cache-sensitive designs, treat **placement** and **parameterization** as separate experiments.

## Combined design rule

MA-241 + MA-253 together suggest the central decomposition:

    function variation
      = compact shared-coordinate component
      + genuinely private residual component

Future screens should estimate the frontier as the fraction of variation shifts from coordinate-aligned to independent/private.

Recommended synthetic interpolation:

    Delta(alpha) = alpha * Delta_view + (1-alpha) * Delta_private

with alpha swept from 0 to 1, while holding actual bytes and compute accounting explicit.

This is more informative than testing only an all-aligned or all-independent teacher.

## Runtime rule

Both experiments found large wall-time overhead from the unfused Givens implementation despite modest analytical MAC overhead.

Until a fused/vectorized implementation exists:
- do not infer runtime from MAC count;
- keep quality/storage and runtime conclusions separate;
- prefer vectorized transforms (diagonal, rank-one, Householder, butterfly, ACDC) as runtime controls.

## What next workers should exploit

1. MA-244/245: separate hard sharing from role/layer recovery; include simple diagonal/rank-one views.
2. MA-247: compare Mirror depth code to static LoRA and generated modulation; include aligned/misaligned depth variation.
3. MA-255/260/265/268: Parameter Superposition, BatchEnsemble, VeRA and IA3 now define stronger low-description controls than ad-hoc gates.
4. MA-319+ tensor-bank candidates: directly test the "shared basis + small coordinate + private residual" decomposition with a richer basis family.
5. MA-332/333: run symmetry audits early so function-preserving reparameterizations are never counted as logical functional multiplicity.
