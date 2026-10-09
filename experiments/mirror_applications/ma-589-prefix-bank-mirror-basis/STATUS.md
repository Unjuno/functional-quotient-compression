# MA-589 status

- Status: FAIL
- Branch: `research/ma-589-prefix-bank-mirror-basis-20261009`
- Base commit: `f6bdb727`
- Evidence commit: `3631003b`
- Source/protocol frozen and amended before registered dev; see `FREEZE.json` and `AMENDMENTS.md`.
- Development complete: yes (58901/58902; same-seed replay exact for core metrics)
- Fresh/audit opened: no (quality and native-alias gates failed)
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Commit verified FAIL and continue with next P0, MA-591.

## Blockers

None.

## Decisions / rulings

This experiment measures text-derived KV prefix-cache compression, not trained continuous Prefix-Tuning. The first failed seed invocation is excluded; one source amendment corrected nested basis indexing before registered results were retained.
