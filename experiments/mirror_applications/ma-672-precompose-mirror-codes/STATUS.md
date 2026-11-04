# MA-672 status

- Status: FAIL (Mirror-specific claim)
- Branch: `research/ma-672-precompose-mirror-codes-20261104`
- Base: `0d97988d`; frozen protocol amended before rerun; fresh remained sealed

## H — Hypothesis

For aligned low-rank LoRA experts mixed by input-dependent gates, precomposing expert codes before applying the shared adapter should preserve output while reducing active adapter work versus executing every expert. Mirror must beat native coefficient-space precomposition to support a Mirror-specific claim.

## T — Treatment and controls

Synthetic one-layer D=32, E=16, rank-4 atoms, 512 examples, residual RMS 0/.05, development seeds 67201/67202. Compared full independent expert matrices, top-2 full-bank reference, shared atom-code precomposition, and native coefficient-space precomposition. Measured compressed NPZ inference payload bytes, output NRMSE, MAC proxy, and CPU inference wall time. The initial attempt had an einsum axis bug and incorrect active MAC accounting; it is preserved in `source/attempts/initial_axis_bug/` and disclosed by protocol amendment before the corrected rerun.

## D — Decision

FAIL for Mirror-specific advantage. At rho=0, corrected precomposition had NRMSE <5e-8 and used 8,029–8,057B versus 67,384–67,427B full bank, with active MAC proxy 950,272 versus 9,175,040. At rho=.05, NRMSE was .010–.012. The native coefficient control had exactly equal outputs and compute; its files were 14B larger solely from serialized key names. Thus precomposition is a strong generic low-rank efficiency result for this synthetic aligned bank, but the Mirror coordinate brings no distinct function or compute benefit. Fresh was sealed at the exact alias gate.

## C — Strongest counter-hypothesis

Linearity of LoRA updates makes gate-weighted adapter addition commute with matrix multiplication. Native coefficient precomposition is algebraically identical and exhausts the observed gain.

## U — Unconfirmed

Natural trained adapters, learned router quality, real accelerator kernels, latency under batching, and residuals beyond rho=.05 remain untested. The estimated MAC reduction is not a measured hardware speedup.

## Evidence labels

- Fact: full-bank and precomposed output errors, payload sizes, and MAC proxies are in `RESULTS_CORE.csv` and `source/development.json`; native and Mirror coordinates compute identical functions.
- Interpretation: precomposition can reduce work when many aligned LoRAs are linearly mixed, but ordinary coefficient algebra explains the reduction.
- Hypothesis: a Mirror-specific advantage would require nonlinearity or structured views that make native coefficient aggregation more expensive or less expressive.
