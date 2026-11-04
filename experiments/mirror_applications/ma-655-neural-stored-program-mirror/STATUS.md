# MA-655 status — complete

- Decision: **FAIL (Mirror-specific claim)**. The screen demonstrates low-dimensional program-memory compression on these synthetic tasks, but Mirror is exactly the native PCA control and is slightly larger than explicit NSPM.
- Protocol: frozen before development at `4525c54538f3d767002b8f8d50d74996b63bbeb4d0d115d1ac7fe97fa9ce1ded`; one development-only correction amended the protocol before fresh access to charge a basis-coordinate transform for unseen runtime codes.
- Development: seeds 65501/65502. Fresh: seeds 65511/65512/65513. Fresh results remain separate in `source/fresh.json`.
- Controls: independent full matrices, explicit four-controller NSPM, rank-4 Mirror/PCA, byte-identical native PCA, and one-hot nearest program.
- Payload: actual compressed `.npz` inference payload, including program IDs, codes, learned basis and the unseen-code coordinate transform. Each row records `stat().st_size`.
- All five methods were measured on every seed × program-count (16/64) × private-residual (0/.05/.2) condition. No optimizer updates; this is a synthetic function-storage screen.
- Initial implementation gave the unseen codes in the generating coordinates directly to the learned PCA decoder. This bug was found on development seeds, preserved in `source/attempts/initial_untransformed_codes/`, corrected with a charged transform, and all fresh seeds were run only after that correction.

## H / T / D / C / U

- **H — Hypothesis:** four shared controller functions plus small per-program coordinates can store many task-useful functions and generalize to unseen coordinates in less memory than independent programs; private residuals mark the boundary.
- **T — Trial:** 32×32 linear programs; 16 or 64 stored programs; 16 unseen codes; residual RMS 0/.05/.2; development seeds 65501/65502 and fresh seeds 65511/65512/65513. Compared independent matrices, explicit NSPM, Mirror rank-4 PCA, identical native PCA, and one-hot retrieval. Quality is held-out output NRMSE; storage is actual serialized bytes.
- **D — Decision:** FAIL for a Mirror-specific advantage. At zero residual, both Mirror and native PCA reconstruct stored and unseen functions to numerical precision, at 16.66 KB vs 61.54 KB independent (N=16), and 17.47 KB vs 244.48 KB (N=64). However, Mirror and native PCA payloads and decoded arrays are exactly equal. Explicit NSPM is also exact on the generated task and smaller (16.38 KB and 17.18 KB). At nonzero residual, held-out errors rise similarly for Mirror/native PCA and remain below .02 only for N=64, rho=.05; NSPM retains unseen-code quality but loses residual variation. One-hot errors are about .54–.63.
- **C — Strongest counter-hypothesis:** the observed compression comes from known rank-4 shared function structure and ordinary NSPM/PCA coding, not an additional Mirror degree of freedom. The synthetic tasks are generated from four shared basis controllers, favoring shared-basis methods by construction.
- **U — Unconfirmed:** natural neural controllers, learned/retrieved programs, online memory updates, larger nonlinear program families, and optimized runtime/latency. The MAC and wall-time fields are proxies from a tiny synthetic screen, not deployment measurements.

## Evidence labels

- **Fact:** at rho=0, all 3 fresh seeds have near-zero NRMSE for Mirror/PCA and explicit NSPM at both N values; Mirror/native PCA arrays are byte-for-byte equal. Fresh mean payloads: N=16 independent 61,543 B, Mirror/native 16,661 B, NSPM 16,377 B; N=64 independent 244,469 B, Mirror/native 17,476 B, NSPM 17,180 B.
- **Interpretation:** the function family has strong rank-4 structure; ordinary shared-basis program memory compresses the bank. The measured bytes do not support a Mirror-specific increment.
- **Hypothesis:** any practical gain will require a task family where learned function coordinates outperform explicit NSPM or native low-rank program codes on held-out programs while charging coordinate transforms and runtime costs.
