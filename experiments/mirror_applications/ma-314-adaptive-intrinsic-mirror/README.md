# MA-314 — adaptive intrinsic dimension with Mirror views

Status: PROMISING only for the aligned coordinate-only variant; broader private-fallback screen missed its byte gate and remains fresh-locked.
Branch: `research/ma-314-adaptive-intrinsic-mirror-20261008`
Prior art: PA33; direct adaptive random-subspace coefficients are mandatory.

## H — hypothesis

On a task bank with different intrinsic complexities, validation-selected dimension plus a Mirror angle for a shared-radius first coordinate pair and private residuals should reduce actual serialized bytes by at least 10% versus validation-selected ordinary intrinsic coefficients at held-out normalized MSE <=1e-4. Tasks outside the orbit should require private intrinsic fallback.

## T — protocol frozen before fresh

A shared 32D linear predictor has one deterministic 32x16 random orthonormal update basis. The 160-task bank includes four aligned groups of 32 tasks at true dimensions 2, 4, 8 and 16, plus 32 unrelated 16D tasks. Each has 64 support, 64 validation and 128 held-out examples. Controls are hard tying, fixed direct dimensions 2/4/8/16, adaptive direct intrinsic coefficients, adaptive Mirror angle plus residual/private fallback, and independent full 32D fits. Optimizer updates: zero. Mirror angle search: fixed 720-point grid. All serialized inference bytes include used basis prefix, shared base/radius, per-task dimensions/modes, offsets, coefficients and archive metadata.

Development seeds 31401/31402 compared validation thresholds 1e-4, 1e-3 and 1e-2. The selected threshold is 1e-4. Before fresh access, the selection rule was clarified to use group-median dimensions, and the active operation proxy was corrected; the initial summaries are retained under `protocol_variants/`. See `PROTOCOL_AMENDMENTS.md`. Fresh seeds 31411–31413 remain locked.

## Current development observation

At 1e-4 both methods chose median dimensions 2/4/8/16 for the aligned groups. Adaptive direct payloads were 6,730/6,734B; adaptive Mirror payloads were 7,082/7,084B. Mirror had 32 unrelated private fallbacks per seed. Development therefore misses the byte gate, but fresh evaluation remains locked and the preregistered fresh worlds are retained to check replication.

## U

No fresh result is available yet. The task bank is synthetic and post-fit; this is not language-model or capacity evidence.

## Protocol reconciliation — separate completed aligned-only variant

An independent MA-314 variant was completed on the same research branch and is retained under [`protocol_variants/aligned_coordinate_only/README.md`](protocol_variants/aligned_coordinate_only/README.md). It uses a 16D predictor, a random 16×8 basis, 48 tasks, and only nested fixed-radius aligned task functions; it has no unrelated-task private fallback. Its own preregistered development gates passed, then its distinct fresh worlds passed 3/3: exact 2/4/8 dimensions, 1,996B versus 2,220B adaptive FP16 coefficients, normalized test MSE 1.20e-6–2.09e-6. Its fit proxy was 371.6M versus 286k operations.

This variant does **not** replace or pool with the broader root protocol above. In the broader 160-task protocol, Mirror plus private fallback quality passed but its 7,082–7,084B payload exceeded adaptive direct's 6,730–6,734B in both development seeds, missing the registered storage gate; fresh seeds 31411–31413 therefore remain unopened. The broad result marks the private-fallback/aggregate storage boundary, while the aligned-only variant is a narrow storage/quality point. Their similarly named seeds come from different world generators and are not paired replications.

**Reconciled scope:** PROMISING only for the completed aligned-only coordinate variant; the broader private-fallback allocation screen failed its development byte gate. Neither result establishes general capacity or natural-task transfer.
