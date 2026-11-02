# MA-565 status

- Status: **FAIL under frozen total-payload gate; scoped depth-quality recovery signal**
- Branch: `research/ma-565-layerdrop-depth-views-20261102`
- Protocol freeze: `7d777ecc`; fresh seeds: 56511–56513

## H / T / D / C / U

**H:** A compact affine depth-conditioned Givens view on a shared four-block LayerDrop-compatible network recovers heldout shallow-depth functions with lower bytes than independent depth maps and beats a static rank-one correction.

**T:** Synthetic 12-channel nonlinear residual network; depths 1 and 4 fit, depths 2 and 3 held out; two development and three fresh seeds; native prefix, static per-depth rank-one output residual, affine Mirror Givens code, and independent effective maps; zero optimizer updates.

**D — FAIL overall, with quality signal.** Fresh mean/max NRMSE: Mirror `1e-15/3.7e-15`, native `0.0430/0.0900`, rank-one `0.0438/0.0900`, independent `9.7e-17/2e-16`. Mirror quality is essentially exact and beats the static rank-one control, but complete payload is `7529 B` vs `12695 B` independent (59.3%), missing the preregistered <=50% gate.

**C:** The teacher is deliberately an affine Givens angle orbit and therefore strongly favors the Mirror code. A richer static low-rank correction could potentially recover the same functions; no trained LayerDrop model is involved.

**U:** LayerDrop-trained Transformer/NLL, natural tasks, GPU latency/energy, true deployment timing, and generic low-rank controls with enough parameters to reach equal quality.

## Facts

- Fresh heldout Mirror error was below `3.7e-15` across all worlds/depths; native max error was `0.0900`.
- Actual total inference payload: Mirror `7529 B`, native `7320 B`, rank-one `8092 B`, independent `12695 B`.
- Mirror is 2.9% larger than native shared-only payload and 40.7% smaller than independent, below required 50% savings.
- Zero optimizer updates. Fit wall time and payload hashes recorded; GPU/energy and inference timing not measured.

## Interpretation

The depth code restores the intentionally aligned view function at unseen depths and beats the tested rank-one residual in quality. It costs more than native and misses the frozen independent-byte threshold. This remains a synthetic mechanism result, not LayerDrop language-model evidence or capacity.
