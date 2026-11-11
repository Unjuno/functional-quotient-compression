# MA-864 status

- Status: FAIL (quality/storage frontier and Mirror-specific attribution)
- Branch: research/ma-864-off-manifold-private-residual-20261111
- Base commit: b6813a75
- Development complete: yes; base LR .1, rank3 .01, rank6 .5, random3 .01, private .01, full .01
- Fresh/audit opened: yes; seeds 311, 331, 347
- Results committed: pending
- Verification committed: pending

## Decision

Private rank-1 improves over shared rank3 and rank6 across fresh runs, but remains 13.3–32.8 pp below full fine-tuning. Its shared-basis payload 41,997 B is 3.52× the 11,933 B full checkpoint. Mirror/native coefficients are identical.

## Next action

Continue registry order with MA-865, a common task-coordinate manifold tournament.
