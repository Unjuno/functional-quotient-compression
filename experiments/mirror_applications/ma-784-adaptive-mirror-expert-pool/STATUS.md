# MA-784 status

- Status: SCREENING — serializer-only amendment; identical-protocol rerun pending
- Branch: `research/ma-784-adaptive-mirror-pool-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Last verified code/protocol commit: pending amendment commit (original freeze: `0b63f8e`)
- Development complete: no (one 1,200-update dense attempt ended at serialization; no metrics/checkpoint retained)
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Rerun the frozen two-seed screen with corrected metadata serialization; do not open the audit span unless all development gates pass.

## Blockers

None identified. CPU-only is sufficient for the small character-LM screen; runtime is measured as an outcome.

## Decisions / rulings

- Draw18 was replayed from the frozen 549-row pool at uniform index 269 with seed `89253d001af66ade263038b97d2e3e75167e1f8cbc9b91e61540174cf75f51cf`.
- Seven conditions and gates are frozen in `PROTOCOL.json`.
- Four preflight tests pass, including full forward/backward for all conditions, Mirror norm/gradient/effect, and one-update serialization reload. No real corpus or audit values were accessed.
- The public corpus has been acquired; the development script only reads the first 90% for train/dev. The audit span remains unopened.
- Attempt 1 (seed 78401, dense) completed 1,200 updates then failed to serialize because the no-router baseline was incorrectly indexed into the NormRouter table. No metric rows or checkpoint were retained; the precise failure is in `source/attempt_log.json`.
- The protocol amendment changes only dense payload metadata (`normrouter_c=null`, `routing=none`) and was made before rerunning. No scientific settings or gates changed.
- No training data, model values, or audit values have been accessed on this branch.
