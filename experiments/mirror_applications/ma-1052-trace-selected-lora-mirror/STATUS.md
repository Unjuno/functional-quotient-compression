# MA-1052 status

- Status: FAIL (scoped to the preregistered Mirror quality gate)
- Branch: `research/ma-1052-trace-selected-lora-mirror-20261125`
- Base commit: `26d91407d`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes, seed 105202
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Result

Canonical serialized-payload audit MASE: Mirror 1.063280; hard-shared TRACE 1.063251; diagonal rank gain 1.061715. Mirror failed the 2% gate and added 543 B over hard sharing. Four independent TRACE roles had audit macro MASE 1.083612 and 46,230,674 B adaptation state, with one role (ETTh2-H192) requiring more private capacity under the 5% per-role criterion.

Pooled in-memory metrics differ from reloaded-payload metrics. Exact repeated replay of the serialized payload was checked; the discrepancy is preserved in `runs/{dev,audit}/replay_verification.json` and the serialized payload metrics are canonical for byte/quality comparison.

## Next action

Update the registry, claim ledger, status board, and worker queue to FAIL; then continue sequentially to MA-1053.

## Blockers

No blocker to this MA. Broader TRACE reproduction is limited by CPU-only hardware and absent official TRACE source; this scoped paper-derived screen completed. No GPU runtime claim is made.

## Decisions / rulings

- Protocol amendment AM-01 fixed coordinate count and per-role update budget after synthetic CPU profiling and before any task training.
- Protocol amendment AM-02 corrected the development window count to match the executed selector before audit.
- Canonical quality is replayed from the serialized inference state. Initial in-memory/reloaded differences for pooled conditions are reported rather than hidden.
