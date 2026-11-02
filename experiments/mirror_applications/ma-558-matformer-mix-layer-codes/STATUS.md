# MA-558 status

- Status: FAIL (Mirror-specific gate)
- Branch: `research/ma-558-matformer-mix-layer-codes-20261009`
- Base commit: `ae62f3aa`
- Last verified commit: `0ff3df2f`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: yes
- Verification committed: yes

## Next action

Development-only FAIL audited and integrated; 2 tests pass; fresh remains sealed.

## Blockers

None.

## Decisions / rulings

Across two development seeds, the factorized Mirror reconstruction exactly recovers six held-out mixed width assignments at 1364B versus the flat 27-configuration table at 27564B. Native prefixes use 728B but miss held-out functions (nMSE .341 and .798). Ordinary direct factorized coefficients are exactly the same representation and payload as Mirror. This is a strong aligned factorization result, but Mirror-specific value FAILS; fresh remains sealed.
