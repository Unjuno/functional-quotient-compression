# MA-546 — Activation View symmetry audit

Status: **NOT ESTABLISHED as a Mirror application; exact gauge invariance established on the fixed linear screen.**

## H — Falsifiable hypothesis
For a two-layer residual linear map, invertibly changing hidden coordinates while transforming adjacent input/output maps by G and G^-1 preserves the same input-output function. This symmetry orbit is not logical multiplicity. An uncompensated transform changes the function, but may be no better or smaller than ordinary changed weights.

## T — Frozen mechanism screen
A deterministic 32→64→32 FP32 residual block; 256 Gaussian input vectors per seed; two development seeds (54601–54602), three fresh seeds (54611–54613), and zero optimizer updates. Tested permutation/sign, positive diagonal scaling, and dense orthogonal QR transforms. Every transformed map, G, G^-1 and metadata is charged in actual uncompressed NPZ bytes. Fresh seeds opened after both development seeds passed the frozen invariance and nontrivial-unpaired-change gate.

## D — NOT ESTABLISHED as Mirror capacity

### Facts
- Paired coordinate changes preserved outputs on all five seeds: max absolute error 0.60–1.43e-6; relative RMS 1.03–1.77e-7. Diagonal condition numbers were 3.42–3.98; permutation/sign and orthogonal transforms were approximately 1.
- Unpaired transform changed outputs: relative RMS 0.249–0.310 for diagonal scale and 0.757–0.818 for permutation/sign or orthogonal transforms.
- Baseline payload was 17,390 B. Paired explicit G+inverse payloads were 51,216 B; unpaired reference payloads 51,220 B. Hashes and sizes are recorded in VERIFICATION.json.
- Two unit tests pass; zero optimizer updates. Baseline cost is 4,096 MAC/input; transformed inference cost was not benchmarked because maps are materialized offline. Serialization timing is recorded per artifact.

### Interpretation
Paired transforms are hidden-coordinate gauge symmetries and must not be counted as extra functions. The unpaired transform is simply changed weight state; it costs more than the baseline and has no task-quality evidence. This supports a symmetry audit rule, not a useful Mirror compression result.

### C — Strongest counter-hypothesis
The exact linear identities may not characterize trained nonlinear Transformers, normalization, attention heads, or activation distributions. This screen is also too simple to establish useful Mirror application.

### U — Still unknown
Whether trained-transformer Views have non-gauge functional changes, and whether any such change improves actual bytes, quality or compute over ordinary parameter transforms, remains unknown.
