# MA-464 status

- Status: SCREENING — protocol frozen; preflight passed before corpus access
- Branch: `research/ma-464-adamix-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Last verified code/protocol commit: pending pre-data freeze
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Commit and push the frozen protocol and implementation, then acquire the two corpora and run the development screen.

## Blockers

None identified. CPU-only is adequate for the small character model; throughput and wall time remain measured outcomes.

## Decisions / rulings

- Draw19 selected MA-464 uniformly from a frozen 542-row eligible pool at index 78. Replay source is `source/random_draw.json`.
- PA83 establishes AdaMix's stochastic adapter mixture and merged inference as the closest native baseline. This screen implements a small nanoGPT-scale counterpart, not the source paper's large-model evaluation.
- Four preflight tests passed, including adapter gradient and exact fold/merge checks.
- No corpus values or model weights have been accessed on this branch.
