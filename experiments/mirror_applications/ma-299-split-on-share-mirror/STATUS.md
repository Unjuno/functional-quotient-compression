# MA-299 status

- Status: FAIL for Mirror-specific advantage
- Source branch: `research/ma-299-split-on-share-mirror-20261008`; reconciled branch: `research/ma-299-split-on-share-reconciled-20261008`
- Base commit: `28194ea`
- Development complete: yes; threshold 0.01 selected from validation
- Fresh/audit opened: yes; final seeds 29921–29923
- Results committed: yes (`0908ca2`)
- Verification committed: yes
- Registry row updated: yes (cumulative reconciliation commit)
- Source result commit: `0908ca2`; latest source verification provenance: `6380945`
- Corrected deterministic fresh metric/byte/compute/hash replay rerun on cumulative branch: 21/21 rows match; timings remeasured

## Result

- Mirror split: 2,337 B, mean normalized held-out MSE 0.000597, max 0.004071; four views, one private split.
- Two-coefficient control: 2,352 B, mean MSE 0.000596; same allocation pattern. Mirror saves 15 B total (0.64%) and uses 1.74× the fit-compute proxy.
- Native no-view split: 6,414 B; independent full: 6,417 B.
- Always-Mirror without fallback: 1,318 B but mean error 0.300.

## Verification

`python -m unittest discover -s experiments/mirror_applications/ma-299-split-on-share-mirror/tests -v`: 5 passed. All 21 final fresh rows replayed exactly for MSE, payload bytes and SHA256.

## Next action

Stop this SETA shared/private allocation family after two consecutive P0 FAIL results with the same Mirror-vs-ordinary-coefficient structural explanation. Redesign required before MA-298/300. This is the repository stop condition; do not infer the result for other families.

## Blockers

None for this fixed NumPy mechanism screen.
