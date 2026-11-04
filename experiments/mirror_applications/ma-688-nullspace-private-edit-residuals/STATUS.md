# MA-688 status

- Status: FAIL (Mirror-specific attribution); private allocation frontier established in this synthetic setting
- Branch: `research/ma-688-nullspace-private-edit-residuals-20261104`
- Protocol frozen before development; compute amendment and corrected rerun recorded
- Development: seeds 68801/68802; rho 0/.05/.2; train edits 24, held-out edits 8
- Fresh/audit: sealed after exact native shared/private alias

## H — Hypothesis

A rank-4 shared edit basis fit on training edits plus per-edit paid null-space residuals can recover held-out efficacy/locality at lower complete bytes than projected full edits. Both shared and private components must be null-space constrained; native controls must be matched.

## T — Treatment and controls

Synthetic 32×32 linear edits, 8 protected keys, 32 edits, rank-4 basis, private residual ranks 0/1/2/4, two seeds. Basis fit uses 24 train edits; held-out evaluation uses 8 edits ×32 inputs. Compared projected full edits, shared-only PCA, shared plus private residual factors, and native PCA plus private factors. Actual NPZ bytes, held-out response NRMSE, protected-key leakage, reconstruction plus application MAC proxy, inference wall time, and serialization wall time were recorded. Initial MAC proxy omitted reconstructing the shared matrices from serialized state; that run is preserved under `source/attempts/initial_undercharged_shared_compute/` and the corrected protocol includes the full one-time bank decode.

## D — Decision

FAIL for Mirror-specific attribution. At rho=0, q0 is exact with leakage <5e-8 and ~17.4KB versus ~123.6KB projected full. At rho=.05, q0 error is .063–.071; q2 reaches .0016–.0046 at ~33.2KB. At rho=.2, q2 error is .029–.053 and misses .02; q4 reaches .011–.018 at ~47KB. All leakage stays below 6e-8 because shared and private factors are both null-space constrained. Native PCA/shared-private controls match outputs and MAC proxy exactly, with only a 14B field-name difference. Decode-inclusive MAC proxy is higher than projected full (393–524k vs 262k for the held-out batch), so the storage/quality frontier has a compute cost. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Ordinary PCA plus per-edit low-rank residual allocation in the protected null space explains the complete result. Mirror adds no function beyond native shared/private factors.

## U — Unconfirmed

No learned model editor, real LOKI optimizer, semantic edit success, adaptive residual allocator, or hardware latency is tested. The teacher is synthetic and deliberately aligned; encode/factorization training cost is not a deployed editor measurement.

## Evidence labels

- Fact: q0 suffices at rho0; q2 is the smallest tested passing rank at rho=.05; q4 is the smallest tested passing rank at rho=.2. All native outputs replay exactly.
- Interpretation: private bytes recover off-basis quality while preserving protected-key locality, but decode compute rises and ordinary PCA-plus-residual fully explains the frontier.
- Hypothesis: adaptive shared/private allocation may improve natural edit storage, subject to matched native PCA/residual controls and measured serving cost.
