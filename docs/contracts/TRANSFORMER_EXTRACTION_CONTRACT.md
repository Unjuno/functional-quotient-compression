# Transformer Extraction Contract

Canonical reconstruction of the D57 architecture-aware extraction constraints.

The purpose is to prevent a mathematically valid quotient for one computation graph from being silently applied to a different Transformer architecture.

## 1. Three extraction layers

### A. Raw storage inventory

Record every relevant parameter/buffer, unique storage group, tie/alias relation, dtype, and baseline-inclusion rule.

### B. Functional module descriptor

Freeze architecture facts that affect exact functional equivalence:

- number of query heads and KV heads;
- GQA/MQA routing map;
- positional operator (including RoPE conventions);
- QK normalization;
- Q/K/V/O bias presence;
- score scale/bias;
- value-space transformations;
- MLP activation/gating;
- normalization type;
- residual-junction conventions.

### C. Derived analysis primitives

Construct gauge-reduced products/operators only after A and B are known. Derived analysis objects are not automatically zero-bit codec state.

## 2. Plain biased dot-product attention

For row-vector convention:

`q(x) = x W_Q + b_Q`

`k(y) = y W_K + b_K`

then:

`q(x) k(y)^T = x P y^T + x a + c y^T + s`

with:

- `P = W_Q W_K^T`;
- `a = W_Q b_K^T`;
- `c = b_Q W_K^T`;
- `s = b_Q b_K^T`.

Therefore `P` alone is complete only when the relevant bias terms vanish.

For the plain bilinear path, an invertible factor transform

`W_Q -> W_Q A`, `W_K -> W_K A^{-T}`

(with corresponding bias transforms) preserves the score. This is a functional gauge, not a rate saving by itself.

## 3. RoPE restricts QK gauge

With a fixed intervening relative positional operator `M_ij = R_i R_j^T`, arbitrary head-space `GL(d)` gauge is not generally valid.

A factor transform must commute with every relevant `M_ij`; the valid gauge is the common centralizer of the positional-operator family.

Therefore a naive universal quotient to only `P = W_Q W_K^T` can discard functional information under RoPE.

## 4. Exact RoPE operator profiles

For pairwise 2-D rotary blocks, the relative operator can be decomposed into fixed projectors/quarter-turn generators with cosine/sine scalar profiles. Consequently `W_Q R_delta W_K^T` decomposes into corresponding operator atoms.

This is an exact analysis representation under the stated convention.

**Boundary:** a small number of scalar profiles does not imply a small codec rate; the operator atoms can still be expensive.

## 5. Nonlinear QK normalization

If Q/K are nonlinearly normalized in head space (for example RMSNorm-like QK normalization), the score is generally not representable by a single fixed bilinear matrix `P` over all inputs.

Preserve factor-level projections and norm parameters unless a separate exact norm-aware reduction is proved.

## 6. GQA / MQA

For query head `h`, record the explicit mapping `g(h)` to its KV group.

- QK uses the query-head projection and the mapped KV-key projection;
- value/output structure uses the mapped shared V projection and the query-head output block.

Shared KV storage may be charged once, but functional references remain distinct.

## 7. Value/output factor gauge

When no intervening V-space operator exists, the product `R = W_V W_O` is invariant to invertible factor changes `W_V -> W_V B`, `W_O -> B^{-1} W_O`.

If the architecture transforms V in head space, this must be re-audited.

## 8. MLP symmetries are activation-specific

- consistent hidden-channel permutations are exact for elementwise activations;
- positive continuous rescaling is exact for appropriate positively homogeneous paths (e.g. ReLU-like), not generic GELU/SiLU;
- gated MLPs such as SwiGLU have only specific coupled rescaling freedoms; do not assume an arbitrary hidden-basis gauge.

## 9. Residual junctions and normalization

Independent basis changes cannot be applied freely to residual branches that are added in a common coordinate system unless the transform is propagated consistently through every connected map.

Normalization parameters and architecture-specific normalization behavior are proof-relevant.

## 10. Tied storage vs functional role

Tied physical storage is counted once in the baseline if the baseline stores it once. Distinct uses in the computation graph remain distinct functional references.

## 11. Extraction acceptance rule

An extracted primitive is valid only when:

1. its defining architecture flags are frozen;
2. its algebraic identity is exact or its approximation error is explicitly bounded;
3. source tensor orientation is unambiguous;
4. bias/norm/position operators are handled correctly;
5. the primitive's codec availability is separately classified as PAID, EXTERNAL_FIXED, or DERIVED.

The extraction contract prevents analysis-side simplifications from becoming unjustified compression claims.
