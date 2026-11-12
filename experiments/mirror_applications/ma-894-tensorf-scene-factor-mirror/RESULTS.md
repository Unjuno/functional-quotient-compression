# MA-894 result — scoped synthetic FAIL

## H — hypothesis

When several fields reuse separable spatial atoms, a compact scene code over shared tensor components reconstructs fresh fields from partial coordinate samples with substantially fewer actual serialized bytes than independent dense fields.

## T — execution

- **Fact:** Implemented a rank-4 scalar CP field over an 8×8×8 lattice. Six development fields fit shared X/Y/Z factors; three fresh fields per seed fit scene codes from 50% observed coordinates and score on the held-out 50%.
- **Fact:** Development seeds were 89401 and 89407. Fresh seeds were 89501, 89503 and 89509. The model used 1,000 Adam updates for shared factors and 500 for fresh codes, fixed before fresh evaluation.
- **Fact:** Controls were independent dense fields, the algebraically identical native scene-mode 4D CP parameterization, and PCA over flattened development fields.
- **Fact:** Actual serialized NPZ inference payloads included shared factors plus scored fresh-scene codes, PCA mean/basis plus fresh-scene codes, or full dense fresh fields with metadata. Development-only codes were excluded because they are not needed at inference; see the post-run accounting amendment. Four verification tests passed. Fresh metric replay reproduced all 12 method rows with maximum nRMSE delta 0.0.

## D — FAIL (Mirror-specific claim, this synthetic scope)

| Method | Fresh held-out nRMSE | Payload bytes | MAC proxy / batch |
|---|---:|---:|---:|
| Shared CP named Mirror | 0.030919 / 0.000017 / 0.122319 | 1426 / 1425 / 1424 | 11520 |
| Native scene-mode CP | 0.030919 / 0.000017 / 0.122319 | 1426 / 1425 / 1424 | 11520 |
| PCA flattened fields | 0.000001 / 0.000007 / 0.000001 | 9537 / 9494 / 9567 | 6144 |
| Independent dense | 0 / 0 / 0 | 5154 / 5196 / 5260 | 768 |

One of three shared-CP fresh errors exceeds the preregistered 0.02 limit (seed 89509); the other two pass. Shared CP payloads are about 27.4% of the median independent-dense payload, but the all-seed quality gate fails and native scene-mode CP is exactly the same function/arrays/bytes/MACs. PCA is much more accurate here but stores more bytes than the independent dense payload. The fit/code-fit wall-clock and MAC proxies are recorded in `RESULTS_CORE.csv`. The query timer is excluded from the decision: code review after unblinding found that the independent-dense branch only wrapped/reshaped a zero-copy NumPy array instead of timing an explicit coordinate gather, so query wall-clock comparisons are invalid. This does not affect bytes, quality, or the exact native-CP alias ruling; query-runtime frontier is NOT ESTABLISHED.

## C — strongest counter-hypothesis

The storage reduction comes from ordinary four-way CP factorization over deliberately shared separable fields. “Mirror scene codes” are simply the native scene-mode CP factor. Seed 89509's high error also shows that fitting a fresh coordinate from half the lattice is not robust under the frozen optimization budget.

## U — unconfirmed

- Whether any scene × component View helps real TensoRF radiance fields or novel-view rendering.
- Whether a converged/fixed-byte frontier or different scene distribution changes the result.
- Real-scene PSNR, camera/view generalization, rendering FPS, and temporal/appearance transfer.

## Fact / interpretation / hypothesis boundary

- **Fact:** The synthetic screen failed its quality gate; Mirror and native 4D CP rows exactly match in outputs, bytes and MAC proxy.
- **Interpretation:** This tested coordinate is a relabeling of native scene-mode CP in the chosen setup, not a Mirror-specific compression mechanism.
- **Hypothesis:** A different coordinate insertion or scene distribution may behave differently; the real TensoRF candidate is not ruled out.
