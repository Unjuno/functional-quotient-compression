# MA-713 status

- Status: FAIL (development byte/quality gates missed; audit unopened)
- Branch: `research/ma-713-della-magnitude-delta-basis-20261109`
- Base commit: `d47e0cc7`
- Protocol frozen: yes (`e6f0f8c7`)
- Development complete: yes
- Fresh/audit opened: no
- Results committed: yes (`8b55d516`)
- Verification committed: pending

## Result

Magnitude top-k beats random DARE at equal bytes: NRMSE .754 vs 3.023. Rank1 PCA reduces bytes only 15.5% and worsens error to .905; higher ranks are larger than magnitude sparse state. Native PCA aliases exactly. Audit stayed sealed.

## Next action

Advance to MA-714, Fisher-weighted Mirror code composition.
