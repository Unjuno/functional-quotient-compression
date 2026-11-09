# MA-719 — conflict-aware shared/private Mirror merger

Status: SCREENING

## Hypothesis

H: A shared task-delta basis plus selective private exception coordinates can preserve each task update and pairwise compositions at lower total bytes than storing all dense deltas, with a measurable private-residual threshold as task conflict rises.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` as shared-basis coefficients plus conflict-routed private residual entries, so independently useful task capabilities can be recovered without forcing conflicting directions into one shared merge.

Prior art: RegMean (PA177), Fisher merging (PA178), TIES (PA27). Controls: independent dense delta bank, shared low-rank PCA only, shared PCA plus top-magnitude private residuals, random private residuals at equal count, and a native PCA+top-k residual code with identical outputs.

## Frozen synthetic mechanism protocol

D=512; 16 task deltas; eight basis-fit tasks and eight held-out whole tasks; development seed 71901. The source family contains a shared rank-8 component plus conflict residuals at heterogeneity levels rho={0,.05,.2,.5}; private residuals use task-specific coordinate supports with sign conflicts among tasks. Fit the shared basis using training tasks only. Sweep private exception budgets k={0,8,16,32,64,128,256} entries/task, selecting residual entries by magnitude; compare random k-entry residuals. Evaluate held-out per-task NRMSE and pairwise sum NRMSE. Serialize one complete bank per condition and charge shared bases, codes, private indices/values, request IDs, outputs and metadata.

### Gates

PASS for this screen requires NRMSE <=.05 on both held-out individual and pair tasks, at least 25% actual byte savings vs dense, and a strictly better quality/byte frontier than equal-budget random private residuals and native PCA+top-k. If no budget reaches quality, record FAIL and do not open audit. Exact native code alias means no Mirror-specific value.

## Scope

Synthetic matrix deltas only. No neural model, natural task, language score, real Fisher/RegMean statistics, TIES implementation, or production runtime claim.

## Results — development seed 71901

**FACT:** At rho=0, shared-only reconstruction is exact and the complete bundle is 54,828 B vs dense 69,355 B (21.0% saving, below the 25% gate). At rho=.05, shared-only held-out/pair NRMSE is .672/.719; magnitude residual k=256/512 reduces it to .185/.175 using 70,652 B, already 1.9% larger than dense. At rho=.2 and .5, k=256 residual gives .252/.245 and .261/.252, with bundles 71,043 B and 71,019 B vs dense 69,233 B and 69,399 B. Matched random residual at k=256 is substantially worse: held-out NRMSE .475, .681 and .699 for rho .05/.2/.5. The payload counts all task reconstructions and four pair outputs. The full sweep is in RESULTS_CORE.csv.

**INTERPRETATION:** With this estimated basis and unstructured private components, magnitude routing beats random placement, but no tested point meets both quality and bytes. At zero heterogeneity, shared coding is exact but misses the strict saving gate. Once residual variation appears, useful quality requires more than half of each vector private in this budget sweep, erasing storage gains.

**HYPOTHESIS:** Structured, correlated private directions or a better task-aware shared basis could lower the private threshold. These possibilities remain untested.

**D:** FAIL for the frozen synthetic shared/private Mirror gate; audit unopened.

**C:** Independent Gaussian residuals are deliberately hostile to sharing, and basis estimation uses only eight tasks; natural related deltas may have more structured residuals. Conversely, synthetic aligned base is favorable at rho=0.

**U:** More than 256 private entries/task, natural task deltas, empirical Fisher/RegMean conflict scores, actual TIES behavior, additional seeds and runtime.

### H / T / D / C / U

- **H:** Selectively private residual coordinates recover conflicting capabilities within a useful full-byte frontier.
- **T:** D=512, 16 tasks (8 fit/8 held out), rank-8 basis; rho 0/.05/.2/.5, private budgets 0..256/task, magnitude vs random; dev seed 71901.
- **D:** FAIL. Quality gate is unreachable in the tested private budget at rho>=.05; max private storage exceeds dense. rho0 misses byte gate.
- **C:** Unstructured residuals and an eight-task basis estimate may exaggerate private requirements.
- **U:** Natural-model/fresh evidence, higher private budgets, and runtime.
