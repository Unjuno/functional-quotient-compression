# MA-510 status

- Status: FAIL (development-only registered gate failure; no fresh evaluation)
- Branch: `research/ma-510-conditional-activation-mirror-steering-20261009`
- Base commit: `48990a5a`
- Protocol frozen: yes
- Development complete: yes; 60 rows across 2 worlds × 3 seeds × 2 rho regimes × 5 methods
- Fresh/audit opened: no; stopped by preregistered development failure
- Results committed: yes (`22fb7d42`)
- Verification committed: yes; development artifact replay test passed (1/1)

## Next action

Continue to MA-511; its held-out factor-composition hypothesis is distinct from this noisy cosine-gate screen.

## Blockers

None.

## Decisions / rulings

All methods share the same CAST condition vectors, keys and threshold; only behavior-vector representation changes. This isolates routing errors from Mirror reconstruction errors.

## Decision

- False-trigger rate passed at .0468, but held-out miss rate was .2888 versus <=.05 gate.
- Generic FP16 coefficients used 4,957B / .000214 NRMSE; Mirror used 5,145B / .000357.
- Fresh worlds were not opened because the frozen development gate failed.
