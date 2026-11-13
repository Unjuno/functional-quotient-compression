# MA-996 — MACE-MP shared force field with material Mirror code

Status: **FAIL for Mirror-specific value in the scoped synthetic mechanism screen**. Official MACE-MP transfer remains **NOT ESTABLISHED**.
Branch: `research/ma-996-mace-material-mirror-20261113`

## H / T / D / C / U

**H — Hypothesis:** a compact material coordinate on a shared conservative potential recovers useful material-specific functions with fewer stored bytes than independent coefficients, while beating native low-rank and sparse controls.

**T — Test:** CPU-only synthetic conservative radial pair potentials, with 30 fixed radial coefficients and three chemical species. Development seed 99601 used 12 reference materials × 20 configurations to fit the shared mean/subspace and choose radial width. Width .65 and rank 2 were frozen before fresh worlds 99611/99613/99617. Each fresh world had four target potentials, eight calibration and 24 held-out configurations per target. Compared shared-only, scalar gate, six-coefficient sparse update, rank-2 Mirror, mathematically identical native rank-2 coefficient factorization, and independently fitted full coefficients. Closed-form least squares; zero optimizer updates. Total fit data: 240 development configurations; fresh calibration 32 and held-out 96 per world. Actual serialized inference payloads are the committed `.npz` files; all arrays, bases, codes, indices, and metadata are charged. Fresh-world CPU batch wall times were 1.194s, 0.210s, 1.069s; these are batch totals, not per-method timings and were measured with independent runs launched concurrently, so no runtime win is inferred.

**D — Decision:** **FAIL** for Mirror-specific value. Mirror and native rank-2 had identical outputs/metrics in all fresh worlds. The four-target serialized payload was 2,066 B for each, 16.5% larger than the independent fitted payload (1,774 B); the active MAC proxy was 148 vs 100 per configuration (+48%). The preregistered 25% storage gate fails.

**C — Strongest counter-hypothesis:** the outcome is fully explained by ordinary low-rank coefficient factorization. In the synthetic world, the targets were generated from that rank-2 family, so perfect recovery does not show useful capacity on natural materials.

**U — Unconfirmed:** official MACE-MP weights and MPTrj/material-reference datasets were not available; natural material quality, adaptation sample efficiency, real MACE inference cost, MD stability, and generalization to unseen chemistry remain untested.

## Facts

- Mirror and native rank-2 combined energy/force nRMSE averaged 1.74e-15 in the three fresh worlds, with per-world values ['1.9e-15', '1.72e-15', '1.59e-15'].
- Shared-only averaged 0.438; scalar gate 0.426; sparse update 0.442; independent fit 0.035. Independent fit missed quality in one world (.104), reflecting its same small calibration budget and ill-conditioning.
- Maximum finite-difference force-gradient residual: 9.43e-11; maximum rotation-equivariance residual: 2.36e-16. All 18 serialized payloads round-tripped exactly.
- Per four-target deployment: Mirror/native rank-2 2,066 B; independent fitted table 1,774 B; sparse update 1,738 B; shared-only 1,038 B. The chosen NPZ container overhead is included.

## Interpretation

The conservative-energy insertion preserves the physical derivative relation in this toy model. That does not isolate a Mirror advantage: the code is exactly the native rank-2 coefficient parameterization, costs more bytes than the fitted independent table in the measured serialization, and adds decode MACs. Any success here is a synthetic low-rank learning signal, not increased natural-task capacity.

## Hypothesis for follow-up

MA-997 should test whether shared sparse support plus per-material codes beats native sparsity-promoting adaptation under exact energy-gradient constraints. The family should pause only if that independent mechanism also collapses to a no-advantage native control.
