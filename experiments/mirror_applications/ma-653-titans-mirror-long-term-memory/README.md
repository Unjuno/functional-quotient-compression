# MA-653 — Titans long-term Mirror memory

Status: FAIL. Evidence lane: synthetic long-term memory storage/quality.

## H

Can a shared value basis and per-key coordinates compress writable long-term memory while preserving recall, and where do private residual directions become necessary?

## T

A Titans-inspired linear memory stored 64 key-value items, with the newest 8 distinct keys also held in an exact short-term cache. Sweep target private residual rank 0/2/4/8. Compare full matrix, shared rank4, shared+private factors, identical native low-rank memory, and explicit KV table. Two development seeds. Measure separate long-term/cache/complete bytes, recall MSE/NLL and surprise-triggered writes.

## D

At target residual rank 0, shared rank4 reaches near-zero long-term error at 43% of full complete bytes. At private rank2/4, residual factors restore recall at 55%/62%; rank8 costs about 76%. Shared-only memory errors grow with residual rank and its surprise gate makes repeated writes. Native low-rank state exactly matches Mirror, so there is no Mirror-specific gain and the complete <=50% gate is missed for private-rank conditions. Fresh remained sealed.

## Fact / Interpretation / Hypothesis

- **Fact:** 40 rows; all bytes and metrics replay; tests pass.
- **Interpretation:** Generic low-rank factorization compresses aligned long-term values; private residuals improve recall at an increasing storage cost.
- **Hypothesis:** Larger banks may better amortize a shared basis, subject to native low-rank controls and realistic retrieval.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
