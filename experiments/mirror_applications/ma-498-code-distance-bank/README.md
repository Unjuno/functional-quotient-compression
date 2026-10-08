# MA-498 — Code-distance regularization for a logical function bank

Status: SCREENING — protocol frozen before development runs  
Branch: `research/ma-498-code-distance-mirror-bank-20261008`  
Prior art: PA95 (ECOC / metric learning)

## H — Hypothesis

Learning a separated continuous codebook for eight logical functions can reduce address interference under Gaussian code noise relative to random orthogonal and random spherical codes, while keeping total inference bytes within 5% of a compact raw-ID bank.

## T — Frozen protocol

A fixed bank of eight linear expert functions. Compare compact binary IDs, random orthogonal 8D codes, random spherical 8D codes and 500-step minimum-distance-regularized spherical codes. Add Gaussian noise to addresses at sigma .05/.1/.2/.3/.5. Measure route error, output RMSE, minimum pairwise distance, actual NPZ bytes and decoder operations. All methods pay the expert matrix bank and any explicit codebook. Development seeds are 49801/49802; fresh seeds 49811–49813 are sealed.

## D — Pending development runs

The exact frozen protocol/source hashes and parent commit are recorded in `freeze.json`. Run seeds 49801/49802 only. Fresh seeds 49811–49813 remain sealed unless the frozen gates pass and native attribution is resolved. Metric learning and ECOC are native controls; code distance alone is not Mirror-specific.
