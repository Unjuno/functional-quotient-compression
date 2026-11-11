# MA-865 status

- Status: FAIL (Mirror-specific frontier and preregistered quality/storage gate)
- Branch: research/ma-865-task-coordinate-tournament-20261111
- Base commit: 0b3392c7
- Development complete: yes; base LR .03, task LR .1
- Fresh/audit opened: yes; seeds 401, 419, 433
- Results committed: pending
- Verification committed: pending

## Decision

Rank3 SVD saves only 5.6% bytes and loses 8.5–9.6 pp mean accuracy. Rank4 matches independent experts but costs 51,093 B vs 43,827 B independent storage. Random rank3/4 stays near .84 mean accuracy. Native and Mirror coefficients are identical.

## Next action

Continue outside the paused continuous-chart coordinate lane with MA-866, collective inference admission-policy coordinates.
