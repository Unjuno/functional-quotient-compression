# MA-658 status — scoped FAIL

- Decision: **FAIL for a Mirror-specific gain.** Consolidation can preserve aligned historical linear functions, but ordinary PCA/shared-basis coordinates do exactly the same thing. The frozen full gate also fails against the private rank-2 payload bytes.
- Frozen protocol hash: `9393270fc54e3ddc94aa3c18ce50c4cbcb70226c4264332d2b48e786c0a1e7c3`.
- Development seeds: 65801 and 65802. Fresh seeds 65811–65813 remain sealed because the development Mirror/native alias gate failed.
- Conditions: 32×32 linear task operators; 8 or 32 tasks; consolidation every 8 tasks; residual RMS 0/.05/.2. Every task was replayed on held-out inputs after each consolidation boundary.
- Controls: independent matrices, private rank-2 matrix adapters, rank-4 Mirror coordinates, and native PCA/shared low-rank coordinates.
- Complete serialized `.npz` payload bytes are recorded per method and boundary, including schema, task IDs, base, basis, codes, or per-task factors.

## H / T / D / C / U

- **H — Hypothesis:** fast task Views can be periodically absorbed into a slow shared basis while retaining earlier functions, with lower total bytes than private fast adapters and a strict improvement over native shared low-rank consolidation.
- **T — Trial:** fixed synthetic worlds on seeds 65801/65802, 8/32 tasks, residual RMS 0/.05/.2, and four-task-sized consolidation blocks. Recompute a rank-4 PCA basis from all seen task deltas, remap old task codes, then measure output NRMSE and transition drift for every retained function. Compare independent full matrices, private rank-2 matrices, and exact native PCA codes. Fresh remained locked.
- **D — Decision:** FAIL. At rho=0 and 32 tasks, rank-4 state preserves functions at zero error and uses 16,838 B versus 122,578 B independent. But it is 4.3% larger than the 16,144 B private rank-2 payload (whose quality is poor at NRMSE ~0.89), and Mirror/native PCA states, codes and outputs are identical. At rho=.05/.2, final 32-task mean NRMSE is ~.048/.188 for both Mirror and native PCA; consolidation transition drift is small (~.0005/.0021), but approximation error remains. The exact native alias already rejects Mirror attribution, so fresh seeds stayed sealed.
- **C — Strongest counter-hypothesis:** this is ordinary truncated PCA over a bank of task matrices; recomputing the basis and refitting coefficients preserves old functions when they lie in the retained span. The result does not require a Mirror mechanism.
- **U — Unconfirmed:** learned online consolidation, language-model quality, real test-time memory, task routing, hardware latency, and whether a different function family could make temporal View remapping outperform native basis refitting.

## Fact / Interpretation / Hypothesis

- **Fact:** all 120 development rows have recorded actual payload sizes; tests replayed 40 rho=0 metric rows; Mirror/native arrays are exactly equal for every development condition. For the 32-task rho=0 final boundary, mean payloads are 16,838 B Mirror/native, 16,144 B private rank-2, and 122,578 B independent.
- **Interpretation:** compression and retention are explained by low-rank shared-function structure; residual heterogeneity creates the expected quality loss. The private rank-2 baseline is smaller but does not preserve these functions, so it is a quality/storage point rather than a superior equivalent model.
- **Hypothesis:** temporal consolidation only adds a distinct Mirror capability if it provides a function-preserving remapping or runtime/byte frontier that ordinary basis updates cannot match.
