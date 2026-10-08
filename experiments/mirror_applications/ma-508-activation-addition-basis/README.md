# MA-508 — Activation-addition behavior bank over a shared basis

Status: SCREENING; protocol frozen before development; Amendment 1 corrects the secondary causal metric before canonical reruns  
Branch: `research/ma-508-activation-addition-mirror-basis-20261008`  
Prior art: PA98 (Activation Addition)

## H — Hypothesis

For an aligned synthetic bank of 32 activation-addition functions, a rank-eight basis learned from 24 behaviors plus explicit per-behavior codes will preserve heldout probe outputs at relative RMSE <=.05 while using <=.65x the actual NPZ bytes of an explicit steering-vector bank.

**Mirror insertion:** this experiment adds a per-behavior code `m_b` to a shared activation-addition basis so that 32 hidden-state steering functions can be called without serializing a full 64-value vector for every behavior.

The closest control is native PCA/SVD of the same steering-vector bank with per-behavior coefficients. Explicit Activation Addition vectors fitted from the same support contrasts are the quality/storage upper. The exact protocol/source hashes are in `freeze.json`; the byte gate includes the fixed probe matrix in every payload. Amendment 1 preserves the first two completed runs and changes only code-causality bookkeeping before canonical reruns.
