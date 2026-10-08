# MA-271 result reconciliation

## Fact

Two dedicated branches recorded separate MA-271 protocols:

- `research/ma-271-oft-mirror-task-views-20261008` trained a 16D task-map model on three fresh worlds. It reported 3,309B Mirror versus 7,253B dense OFT and 7,261B independent, with 3/3 aligned quality/storage gates. This protocol did not include the exact simple rank-one shared-angle control.
- `research/ma-271-oft-mirror-views-20261008` ran a four-seed post-fit orbit screen and included that control. Mirror and ordinary rank-one angle factorization were exactly identical in every tested function and both serialized to 1,086B. On aligned tasks their mean MSE was 6.18e-31; on independent plane angles both had MSE 0.1601 while dense/blockwise OFT were near zero. Orthogonality and inverse-cycle errors were <=3.2e-16 and <=2.3e-16.

Both source reports, protocols, raw results and verification artifacts are preserved. The seeds/tasks are not pooled.

## Disposition

MA-271 is **FAIL for Mirror-specific advantage**. Structured orthogonal task coordinates can compress a Givens-aligned family relative to dense OFT, but the stronger simple rank-one control reproduces the Mirror functions and bytes exactly. The first trained screen supports an aligned compression result but does not isolate a Mirror-specific benefit because it omitted that control.

## H / T / D / C / U

- **H:** task-specific orthogonal views over one shared object beat native OFT and the cheapest simple task-code control in quality per serialized byte.
- **T:** separate trained dense-OFT cross-over (fresh 27102–27104; LR .01, 1,000 updates) and post-fit exact-control screen (fresh 127/239/331/443; zero optimizer updates). Both used synthetic, intentionally structured tasks.
- **D:** FAIL for Mirror-specific value; ordinary rank-one task-code × shared-angle parameterization exactly matches the tested Mirror representation.
- **C:** the compact result reflects a known low-rank factorization of the task-by-plane angle matrix, and the first OFT screen used a dense, compute-heavy native parameterization.
- **U:** natural task transforms, nonlinear/diffusion quality, optimized kernels, and whether other Mirror coordinates beat suitable low-rank/OFT controls.
