# MA-508 — Activation-addition behavior bank over a shared basis

Status: **FAIL for Mirror-specific attribution** (rank-8 aligned quality/storage subgate passes)  
Branch: `research/ma-508-activation-addition-mirror-basis-20261008`  
Prior art: PA98 (Activation Addition)

## H — Hypothesis

For an aligned synthetic bank of 32 activation-addition functions, a rank-eight basis learned from 24 behaviors plus explicit per-behavior codes will preserve heldout probe outputs at relative RMSE <=.05 while using <=.65x the actual NPZ bytes of an explicit steering-vector bank.

**Mirror insertion:** this experiment adds a per-behavior code `m_b` to a shared activation-addition basis so that 32 hidden-state steering functions can be called without serializing a full 64-value vector for every behavior.

The closest control is native PCA/SVD of the same steering-vector bank with per-behavior coefficients. Explicit Activation Addition vectors fitted from the same support contrasts are the quality/storage upper. The exact protocol/source hashes are in `freeze.json`. Amendment 1 preserves initial runs and corrects only causal-metric bookkeeping before canonical reruns.

## T — Executed protocol

Two frozen worlds (50801/50802); 32 behaviors; hidden dimension 64; 16 fixed behavior probes. Basis fitting used behavior IDs 0–23. Heldout behavior IDs 24–31 each had 64 support activation contrasts and 256 evaluation inputs. The fixed rank sweep {2,4,8,16} and private-residual sweep rho {0,.1,.25,.5} were evaluated without rank selection. Probe matrix, vectors/codes and schema were charged in actual uncompressed NPZ payloads. Fresh seeds 50811–50813 remain sealed.

## D — FAIL for Mirror-specific attribution

At rank 8 and rho=0, heldout probe relative RMSE was 0 in both seeds and actual payload was 8,324 B versus 13,194 B explicit vectors, a 36.9% saving and ratio 0.631 (within the frozen <=.65 gate). At rho=.1, relative RMSE was 0.0629/0.0799; rho=.25 gave 0.1547/0.1997 and rho=.5 gave 0.2982/0.3885. Rank 2 and 4 were smaller but inaccurate (rho=0 RMSE .867–.987 and .797–.883); rank 16 was exact at rho=0 but cost 11,397 B and 1,088 operation-proxy units per example.

At rank 8, basis decoding plus addition used 576 operation-proxy units per example, nine times explicit vector addition (64). The code changes heldout hidden states by max 0.238–0.285 when zeroed. Native PCA coefficient storage and Mirror have byte-identical payloads and exact output metrics at every rank, rho and seed. Therefore this is a positive standard-PCA compression frontier in the aligned synthetic steering bank, but **FAIL for Mirror-specific attribution**. Fresh remained sealed.

## C — Strongest counter-hypothesis

The aligned directions are generated from a low-rank bank, so ordinary PCA plus coefficients naturally recovers them. The reduced bytes are purchased with reconstruction work at inference; rank 8 costs nine times the explicit addition proxy. Private direction content progressively lowers function quality.

## U — Still unconfirmed

This is not a language model, semantic behavior or human preference evaluation. The fixed random probes and synthetic vectors do not establish on-prompt steering, off-target semantic degradation, task retention, or production latency. The 32 functions are directly evaluated codes, not combinatorial capacity.

## Evidence classification

**Facts:** rank/rho curves, serialized byte lengths and hashes, zero-code causal checks and replay are retained in `runs/`, `RESULTS_CORE.csv` and `VERIFICATION.json`; pre-amendment outputs are preserved separately.  
**Interpretation:** rank-eight PCA compression passes the aligned quality/byte screen but is exactly native PCA, with a compute tradeoff and private-variation quality loss.  
**Hypothesis:** natural Activation Addition banks may be more or less compressible; this experiment does not answer that question.
