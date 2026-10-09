# MA-301 status

- Status: SCREENING
- Branch: `research/ma-301-continuous-mirror-supermask-20261009`
- Base commit: `4046ee8f`
- Development complete: yes (worlds 30100–30101 × seeds 0–2)
- Fresh/audit opened: no
- Protocol locked: after this commit, before fresh

## Next action

Run frozen fresh worlds 30110–30112.

## Development notes

Added fitted rank-2 logistic factorization after development showed PCA alone was too weak. The logistic control uses fixed 250 Adam updates at lr=0.04 and is serialized in fp16. This condition is frozen before fresh evaluation.
