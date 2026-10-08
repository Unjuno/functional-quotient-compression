# MA-1075 status

- Status: FAIL (development screening)
- Branch: `research/ma-1075-retrieval-mirror-20261008`
- Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`
- Development complete: yes, fixed-budget screen
- Fresh/audit opened for quality metrics: no
- Results committed: pending
- Verification committed: pending
- Registry status: FAIL (scoped to registered HSTU/low-rank screen)

## Next action

Preserve the failed screen and continue with a new random draw; do not tune from fresh data.

## Blockers / boundaries

VQ-Rec was not reproduced, so no comparison against its native item-code quality is claimed. MovieLens CPU screen only; one development seed and 16,000 training examples, no capacity claim.

## Decisions

- Development failure: Mirror Givens scoring is below native HSTU and the byte-near pair-gain control on macro genre nDCG@10; interleaved P99 is 11.7% slower than pair-gain.
- No fresh evaluation was opened because the preregistered development stop condition was met.
