# MA-803 — Koopman eigenmode factorization

Status: **FAIL for Mirror-specific value and registered storage gate**

## Prior-art boundary

PA214 motivates stable regime-specific latent operators; PA74 motivates native structured state-space baselines. This screen uses known diagonal modal dynamics under a known orthogonal nonlinear observation map. It does not train S4 or discover Koopman eigenmodes from nonlinear data.

## Fact

Seed 80301; 8 modal dimensions; nine regimes with two diagonal corners held out; 48-step forecasts; 128 train sequences per observed regime and 128 test sequences/regime; no optimizer updates. Independent dense modal operators use 1,723 B and heldout horizon nRMSE 1.01e-7. A shared mean eigenvalue vector uses 1,404 B but nRMSE .02737. Factorized eigenmode codes use 1,695 B, nRMSE 1.01e-7, registered spectral radius .680 and out-of-grid (2,2) radius .630. This is only 1.6% smaller than independent dense matrices, missing the 25% byte gate. A native per-regime diagonal eigenvalue table is smaller at 1,667 B. The native table and factorized View both use 136 active MAC/step vs 192 for dense independent. The Mirror and native spectral factorization payload/hash match exactly. All payload roundtrips are exact.

## Interpretation

Known eigenmode coordinates preserve stability and heldout forecast quality in the synthetic system. However, factorized mode codes fail the registered 25% storage target against dense operators and use 28 B more than the simpler native diagonal eigenvalue table at equal modal MACs. The Mirror code exactly aliases native spectral factorization. This is no Pareto improvement.

## H / T / D / C / U

- **H:** Shared eigenmodes plus compact regime codes recover stable heldout Koopman forecasts at lower complete bytes and compute than independent operators.
- **T:** Seed 80301; 8D diagonal latent dynamics; nine two-factor regimes, two held out; 48-step rollout; dense independent, native diagonal, shared mean, Mirror spectral factorization, native spectral control. Measure bytes/hash, rollout error, registered/out-of-grid spectral radius, MAC proxy, examples, wall calibration and exact replay.
- **D:** **FAIL.** Quality and stability pass, but byte savings versus independent dense are only 1.6%; native diagonal eigenvalues use 28 B less at equal active MACs; native spectral factorization exactly aliases Mirror.
- **C:** The system is diagonal in the known basis by construction. Native diagonal parameter storage is the simplest representation; standard spectral factorization explains the View exactly.
- **U:** Learned eigenspaces, non-normal systems, eigenvalue crossings, trained S4/Deep Koopman, realistic nonlinear dynamics, fresh regimes and fused accelerator kernels.

Fresh remained sealed after the storage gate miss and exact native alias. No trained Koopman/S4 claim is made.
