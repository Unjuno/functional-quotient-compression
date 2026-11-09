# MA-714 — Fisher-weighted Mirror code composition

Status: **FAIL at development; audit unopened**  
Evidence lane: MECHANISM/QUALITY/STORAGE  
Base commit: `6b6d3bc3`

## Hypothesis

H: Projecting Fisher-weighted merge objectives into a shared task-delta basis improves on unweighted code averaging and retains full Fisher merge quality with a smaller complete payload.

## Mirror insertion

> **Mirror insertion:** merge task coordinates using projected per-task Fisher precision, then retain a shared basis and one composed code for each merge request.

PA178 motivates Fisher importance weighting; PA26 supplies Task Arithmetic. Native projected-Fisher solving is the mandatory attribution control.

## Results

**FACT:** Development seed 71401, D=256, eight synthetic task deltas, rank8 basis, 16 merge requests with four tasks each. At private fraction ρ=0, full parameter Fisher payload was 35,416B with normalized Fisher loss .5653. Fisher-weighted code payload was 20,064B (56.6% of full) with loss .7150; unweighted code payload was 11,628B with loss .7211; Task Arithmetic payload was 26,980B with loss .7211. At ρ=.1, Fisher-code loss was .7364 vs full .5789; at ρ=.25, Fisher-code loss .7492 vs full .5760. Mirror and native projected-Fisher code payload hashes and outputs were identical. Mean Fisher-code solve wall time was measured in source results; operation proxy is recorded per output.

**INTERPRETATION:** Code-space Fisher weighting slightly improved the normalized objective over unweighted code averaging at ρ=0, but the change was under 1%, far short of the required 10%, and the loss was 1.265x full Fisher. Storage fell by 43.4%, but quality failed the 1.05x gate. Native projected Fisher fully explains the code solver.

**HYPOTHESIS:** Fisher importance helps choose code-space merge points, but restricting the merged vector to the shared basis sacrifices too much full-parameter Fisher quality in this setting.

**BOUNDARY:** Synthetic diagonal log-normal Fishers and quadratic task deltas only. No empirical Fisher estimates, pretrained models, natural benchmarks, or GPU runtime claims. Audit stayed sealed after development gate misses.

## H/T/D/C/U

- **H:** Fisher-weighted shared code merges preserve quality while reducing state relative to full Fisher merging and unweighted codes.
- **T:** Development seed 71401; D=256, 8 sources, 16 merge requests, 4 tasks/request, rank8, private fractions 0/.1/.25; full Fisher, Task Arithmetic, unweighted and Fisher code controls.
- **D:** **FAIL at development.** Bytes improve, but Fisher-code loss is 1.265x full and under 1% better than unweighted codes; native method is exact.
- **C:** The shared-basis restriction loses directions needed by coordinate-wise Fisher weighting; ordinary projected Fisher code solving explains all results.
- **U:** Whether empirical Fisher statistics and trained task deltas produce a lower-loss code subspace. No audit or natural-task/runtime evidence.
