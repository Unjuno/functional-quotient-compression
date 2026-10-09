# MA-610 status

Stage: complete — **FAIL (development; fresh sealed)**.
Decision: usage pruning beats random four-view selection by ~18× in held-out error, but is >100× worse than full Mirror in both worlds and far worse than four-view-from-start. Actual usage-pruned payload (3,285 B) exceeds overcomplete pool (3,225 B) because retained indices/metadata are charged.
Verification: 16/16 payload size/hash/replay passed; 2 tests passed.
Last verified commit: pending result commit.
