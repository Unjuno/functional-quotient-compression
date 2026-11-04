# MA-669 status — scoped FAIL for Mirror attribution

- Decision: **FAIL for a Mirror-specific gain.** On aligned tasks the learned 2D code is much cheaper to adapt than derivative-free LoRAHub search, but the native low-rank coefficient prior is the exact same function and state; at 16 tasks the complete bytes also miss the LoRAHub-relative gate.
- Protocol was frozen at `b6c9e858` and amended twice before fresh access. Final hash and amendment records are in `freeze.json`.
- Development seeds: 66901/66902. Fresh seeds 66911–66913 remain sealed after exact native low-rank alias.
- Conditions: 8×8 linear regression; eight rank-1 LoRA atoms; 16/64 tasks; support 2/4/8; residual RMS 0/.1/.25; 128 held-out queries/task; 128 disjoint meta tasks for the coefficient subspace.
- Actual payload charges base, all atom factors, meta coefficient basis, task codes/full coefficients, IDs and metadata.

## H / T / D / C / U

- **H — Hypothesis:** a learned 2D task code can replace full-bank LoRAHub coefficient search with fewer support evaluations and less task state while retaining query quality, beyond ordinary native low-rank coefficient priors.
- **T — Trial:** compared independent full matrices, derivative-free Powell search over all eight coefficients (maxfev=3000), direct full coefficient least squares, Mirror 2D code, native low-rank 2D code, and oracle coefficients. Two development worlds; 216 rows across tasks, shots and private coefficient residuals. Fresh remained locked.
- **D — Decision:** FAIL for Mirror attribution. At rho=0 and 4 shots, N=64 Mirror achieved zero held-out NRMSE at 3,478 B versus LoRAHub's 4,828 B and about 1,304 derivative-free objective evaluations/task. At N=16, Mirror used 3,008 B versus 3,338 B LoRAHub (90%, missing the <=75% gate), with zero error for both. Mirror's shared base, atom bank, coefficient basis, latent codes and outputs exactly match the native low-rank control; its actual NPZ is 4 B larger because of field-name metadata. At rho=.1/.25, Mirror low-rank codes lose task-specific residuals (e.g. N=64, 4 shots: NRMSE .114/.278), while full-coefficient fitting/LoRAHub reaches near-zero error on this noiseless support setup. The smaller code is not independent capacity.
- **C — Strongest counter-hypothesis:** the apparent search-efficiency gain is an ordinary low-rank prior over LoRAHub coefficients. Native two-dimensional coefficient coding reproduces the same support solution, quality, and compute; direct least squares is another simpler native control.
- **U — Unconfirmed:** real language-model task transfer, LoRAHub paper-scale coefficient search, noisy few-shot adaptation, different adapter ranks, cross-family transfer of the meta basis, and inference wall time on accelerator hardware. The task distribution is synthetic and deliberately generated from a rank-2 coefficient prior.

## Development history and evidence labels

- **Fact:** initial coordinate search capped at 97 evaluations/task and underfit; its 216 rows, payloads and protocol are preserved in `source/attempts/initial_coordinate_search/`. A development pilot showed Powell at 300 evaluations still left NRMSE .010, while convergence took 1,635 evaluations for one aligned case; the 300-cap results are preserved separately. Final derivative-free Powell search used maxfev=3000 and typically converged after ~1.2–1.4k evaluations/task. Fresh was never accessed.
- **Interpretation:** the 2D learned code is an efficient adaptation parameterization for the deliberately aligned synthetic task family, but its benefits are not Mirror-specific and private coefficient residuals require more task parameters.
- **Hypothesis:** a learned low-rank prior may reduce search compute for real related tasks if the basis transfers, but it must beat direct native coefficient subspaces on held-out task families and full payload bytes.
