# MA-690 status

- Status: SCREENING
- Branch: `research/ma-690-orbit-private-frontier-20261109`
- Base commit: `53040b26`
- Protocol frozen before implementation/run: yes; protocol hash will be recorded with results
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## H — Hypothesis

At fixed actual payload bytes, at least one family-specific View coordinate plus low-rank private residual will reconstruct cross-family operator banks better than residual-only SVD on multiple alpha points and generators.

## T — Treatment and controls

Frozen plan: D=16, 8 tasks/bank, development seeds 69001/69002, seven target generators, six candidate families, alpha amplitude mixture 0/.25/.5/.75/1, residual ranks 0/1/2/4/8/full. Every candidate is evaluated on every generator. Native family coordinates, independent full matrices, backbone-only, and byte-matched residual-only SVD are controls. Givens/BOFT are local tangent screens.

## D — Decision

Pending frozen development runs.

## C — Strongest counter-hypothesis

Residual-only SVD or a standard native parameterization may match the structured View frontier at equal actual bytes and compute.

## U — Unconfirmed

The planned screen does not test learned task adaptation, nonlinear orthogonal finetuning, language quality, downstream generalization or optimized kernels.

## Next action

Implement the frozen local-operator bank, serialized controls and byte-matched SVD frontier; then run only development seeds 69001/69002.
