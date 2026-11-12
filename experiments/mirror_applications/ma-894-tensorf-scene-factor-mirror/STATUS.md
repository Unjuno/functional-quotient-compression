# MA-894 status

- Status: FAIL (scoped synthetic mechanism screen)
- Branch: `research/ma-894-tensorf-scene-factor-mirror-20261112`
- Protocol frozen before fresh evaluation: yes
- Development seeds: 89401, 89407
- Fresh seeds: 89501, 89503, 89509
- Fresh results: recorded in `RESULTS_CORE.csv` and `RESULTS.md`
- Verification: 4 tests passed; deterministic fresh replay max nRMSE difference 0.0
- Registry row updated: yes

## Decision boundary

The synthetic Mirror scene-code formulation is an exact native scene-mode CP alias and misses its all-seed quality gate. No claim is made about real TensoRF scenes, novel-view PSNR, or rendering FPS.

## Next action

Proceed to MA-895 (K-Planes spatial versus temporal Views) after saving this result and verifying registry integrity.
