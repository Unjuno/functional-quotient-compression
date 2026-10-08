# MA-292 status

- Status: FAIL
- Branch: `research/ma-292-task-vector-mirror-basis-20261008`
- Base commit: `e259f27`
- Development complete: yes
- Fresh/audit opened: yes, locked seeds only
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Record checked result and registry/claim-ledger updates, then push the dedicated branch.

## Blockers

None. NumPy/SciPy CPU execution is available; PyTorch/GPU are not needed for this synthetic mechanism screen.

## Decisions / rulings

- FAIL for Mirror-specific value: matched-byte FP16 SVD has lower total storage, slightly better quality, and higher inference throughput in all fresh worlds.
- Random candidate selection and collision check are recorded in PROTOCOL.json.
- The task-vector rank is fixed at two for the registered Mirror view; rank-1 SVD is a control, not a tuning candidate. Rank-2 FP16 SVD is the matched-byte simple control.
