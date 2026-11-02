# MA-550 status

**FAIL for Mirror-specific attribution; the mixed representation mechanism passes its synthetic reconstruction gate.**

- Development: 55001, 55002; fresh: 55011, 55012, 55013.
- All five worlds select eight activation codes for dense functions and eight sparse output biases for sparse functions.
- Adaptive payload: 19,682 B, fresh max per-function relative RMSE 1.05e-4, decode proxy 412.1M FLOPs/bank.
- The selected method exactly aliases native sparse output-bias plus ReFT/shared-output-basis coding. No Mirror-specific improvement is established.
- Tests: 4 passed; development payload replay exact; fresh arrays and payload hashes are recorded.

**Audit note:** `VERIFICATION.json` has contradictory nested `checks.fresh_opened: false` and top-level `fresh_opened: true`. The final result table contains all three fresh worlds and hashes, and the result report describes them as completed. Retain this provenance inconsistency; do not infer extra evidence beyond the checked artifacts.

Scope: synthetic Pythia-70M output-delta allocation only; natural task quality, OOD routing and serving latency remain untested.
