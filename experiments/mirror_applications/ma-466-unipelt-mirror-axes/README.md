# MA-466 — UniPELT components as factorized Mirror axes

Status: **FAIL for Mirror-specific advantage; factorized PEFT composition PROMISING in this synthetic family**
Evidence lane: MULTI PEFT COMPONENTS / FACTOR GATES / ABLATIONS / BYTES
Protocol frozen: `f2255673`; fresh worlds 46610–46612.

## H — Hypothesis

Task, layer, and position factors can compose three logical PEFT component gates with lower bytes than a generic UniPELT-style generator while retaining held-out combinations; each axis should causally matter.

## T — Test

Synthetic nonlinear regression with three component functions: LoRA-like `x₁x₂`, prefix-like constant `1`, and adapter-like `tanh(x₁)`. Targets were weighted sums with rank-one task×layer×position gate coefficients over 6×3×2=36 combinations. Twenty-seven combinations trained the factor model; nine combinations were held out. Compared product-factor Mirror, ordinary CP product, HyperFormer gate MLP, per-combination support-fit gates (UniPELT-style upper control), and each component ablated. Three fresh worlds × three seeds; 64 support and 256 query points per combination. Actual packages include factor tables/gates, combination indices, and component metadata.

## D — FAIL for Mirror-specific claim; scoped factor composition works

**Fact:** At N=36, Mirror and generic CP had identical held-out NRMSE 0.000203 and identical 2,653B payloads (73.69B/combo). HyperFormer gate MLP scored 2.438 at 4,569B; per-combination support-fit gates were exact at 2,469B. Ablating the LoRA, prefix, or adapter component raised NRMSE to 0.571, 0.488, or 0.246 respectively. Each axis is active. Mirror/CP used 9 decoder MACs/combo vs 384 HyperFormer MACs/combo.

**Interpretation:** Factorized component gates compactly recover this structured synthetic family and beat the trained concatenated-ID MLP, but the same construction is ordinary CP factorization with identical payload hashes and outputs. The support-fit gate control is smaller and exact when per-combination adaptation is allowed. No Mirror-specific gain is established.

## C — Strongest counter-hypothesis

The teacher is explicitly rank-one across task, layer, and position. Generic CP is the direct mathematical explanation for the result; the HyperFormer MLP also appears underfit at the fixed update budget.

## U — Unknown

Natural UniPELT/Transformer adaptations, higher-rank component interactions, and training to near-convergence remain untested. This is a synthetic component ablation screen only.
