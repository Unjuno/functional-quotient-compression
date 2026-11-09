# MA-713 — DELLA magnitude-aware Mirror delta basis

Status: **FAIL at development; audit unopened**  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `d47e0cc7`

## Hypothesis

H: Magnitude-aware retention followed by shared-basis task codes preserves task function with fewer complete payload bytes than sparse deltas alone, beyond random DARE and native PCA controls.

## Mirror insertion

> **Mirror insertion:** task-specific magnitude-retained deltas are represented by coefficients over a shared basis and decoded on demand.

PA176 motivates magnitude-aware retention; PA27 requires sign-conflict controls. This screen uses top-magnitude retention as a transparent DELLA-inspired sparsifier, not the full DELLA procedure. Controls are dense deltas, magnitude top-k, random DARE, PCA ranks 1/2/4/8, native PCA, and TIES merge.

## Results

**FACT:** Development seed 71301, D=2048, 16 task deltas, 90% retention, teacher rank 8. At rho=0, magnitude top-k used 20,666B and task-output NRMSE .754; random DARE at the same nonzero count used 20,666B and NRMSE 3.023; dense deltas used 131,250B. Magnitude-PCA rank1 used 17,454B (84.5% of magnitude sparse) and NRMSE .905; ranks2/4/8 used 25,710/42,222/75,246B and NRMSE .826/.768/.729. At rho=.05/.2, rank1 NRMSE was .898/.946 versus magnitude top-k .745/.738. Native PCA hashes and outputs were exact aliases at every rank. TIES trim20 used 8,714B as a single merged model with NRMSE 1.287 at rho=0; it is not a per-task model bank.

**INTERPRETATION:** Magnitude selection substantially outperformed random dropping at equal sparse bytes. PCA rank1 saved only 15.5% over magnitude top-k and degraded quality beyond the frozen +.02 tolerance; ranks2/4/8 cost more bytes than magnitude top-k. No shared-basis point met the preregistered 25% byte reduction and DARE-relative quality gates. Native PCA fully explains the candidate representation.

**HYPOTHESIS:** Magnitude-aware retained values can have more shared structure than randomly retained values, but this screen did not show enough additional basis compression to preserve the magnitude sparsifier’s quality.

**BOUNDARY:** Synthetic low-rank task deltas only. The magnitude top-k comparator is DELLA-inspired, not a full DELLA reproduction. No pretrained fine-tuning, natural benchmark, or GPU runtime claim. Audit stayed sealed after the development gate miss.

## H/T/D/C/U

- **H:** Magnitude-aware selection leaves task deltas more compressible and less interference-prone than random dropping.
- **T:** Development seed 71301; D=2048, 16 tasks, rank8 teacher, rho 0/.05/.2, 90% retained-drop fraction, PCA ranks1/2/4/8, dense/random/TIES controls.
- **D:** **FAIL at development.** Rank1 saves 15.5%, misses the 25% byte gate and loses >.02 quality; higher ranks cost more. Native PCA exactly aliases.
- **C:** Magnitude top-k alone explains most of the quality improvement; shared PCA is a generic lossy approximation.
- **U:** Whether true DELLA sampling or trained model deltas change the frontier. No audit or natural-task claim.
