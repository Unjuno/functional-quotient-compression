# MA-292 status

- Status: SCREENING
- Branch: `research/ma-292-task-vector-mirror-basis-20261008`
- Base commit: `e259f27`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Protocol frozen at eight support vectors; commit the freeze before fresh seeds.

## Blockers

None. NumPy/SciPy CPU execution is available; PyTorch/GPU are not needed for this synthetic mechanism screen.

## Decisions / rulings

- Random candidate selection and collision check are recorded in PROTOCOL.json.
- The task-vector rank is fixed at two for the registered Mirror view; rank-1 SVD is a control, not a tuning candidate. Rank-2 FP16 SVD is the matched-byte simple control.
