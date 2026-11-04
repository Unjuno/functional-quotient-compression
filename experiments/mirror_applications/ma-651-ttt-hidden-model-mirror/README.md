# MA-651 — TTT hidden-model Mirror parameterization

Status: FAIL. Evidence lane: synthetic test-time state quality/storage.

## H

Can a compact code select writable rank-4 hidden-state subspaces across sequence streams while retaining online adaptation quality and reducing actual recurrent-state bytes?

## T

A supervised TTT-like linear learner adapts 16 streams for 128 tokens, with four rank-4 output modes. Compared full TTT-Linear, one fixed basis, private per-stream rank-4 bases, shared dictionary codes, and native IDs. Two development seeds. All bases and writable states were serialized. Development selected common SGD rate .3; fresh remained sealed.

## D

Shared dictionary state is 41.5–41.7% of full TTT-Linear state and matches private rank-4 quality after correct bootstrap routing. Yet private rank-4 state is smaller in actual bytes (6.94–6.99KB vs shared dictionary 7.12–7.18KB), and native IDs exactly match Mirror. Fixed shared basis is smaller but loses about 9–10x on final-window MSE. FAIL for Mirror-specificity.

Two superseded development attempts (wrong basis orientation and unstable LR=1.0) are preserved under `source/attempts/` and documented in `source/attempt_log.json`.

## Fact / Interpretation / Hypothesis

- **Fact:** 10 final development rows; tests, byte verification, and metric replay pass.
- **Interpretation:** Low-rank state helps this aligned task versus full TTT, but codebook sharing is dominated by private low-rank storage and ordinary IDs.
- **Hypothesis:** Larger stream banks may amortize the shared dictionary overhead on natural test-time learning tasks.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
