# LoRA Mirror composition family diagnostic (2026-11-04)

## Decision

Operationally pause the linear shared-LoRA coefficient/factor subfamily represented by MA-676 and MA-677 until a Mirror-specific insertion point is proposed. Continue registry work in other P0 families; MA-679 is next.

## Evidence

- MA-671: shared-basis MoLE function aliases its native shared-basis control. The full-bank comparison showed a compact aligned representation, but the Mirror-specific attribution gate failed.
- MA-672: after correcting runtime and controls, rank-4 code precomposition is exact at ~8KB / 950,272 MAC proxy versus independent rank-4 LoRA at ~21.9KB / 2,883,584. Native coefficient precomposition exactly matches function and MACs. Rank-2 private residual at rho=.05 slightly improves error while doubling shared state and compute.
- MA-673: corrected rank-1 LoRA task×layer product factors exactly match native outer-product factorization and miss the 60% byte gate (~90% of independent bytes).
- MA-674: corrected top-k atom execution loses output quality for k=2/4/8. k=16 is exact but aliases the native low-rank execution; it uses all atoms and provides no Mirror-specific function.

## Interpretation

Across these proposals, the observed compression comes from ordinary low-rank basis sharing, coefficient linearity, and factorized execution. The exact native controls explain the function; no experiment demonstrates a Mirror-specific gain. Sparse selection also exposes a quality/compute tradeoff rather than independent logical capacity.

## Resume condition

Do not run MA-676 as ordinary trained-dictionary pruning, or MA-677 as shared atoms plus private residual only. Resume when a preregisterable Mirror mechanism differs from native learned-dictionary pruning, random pruning, small-from-start, and standard residual allocation. Keep those registry rows UNTESTED and document this operational pause in their worker notes.
