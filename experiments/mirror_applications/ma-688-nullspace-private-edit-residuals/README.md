# MA-688 — Null-space private edit residual frontier

Status: FAIL (Mirror-specific attribution); synthetic shared/private frontier measured  
Evidence lane: MECHANISM / STORAGE / COMPUTE  
Base commit: `fa73528c`

## Hypothesis

H: A shared rank-4 edit basis plus a small per-edit null-space residual can preserve held-out edit efficacy and protected-key locality with lower complete serialized bytes than projected full edits.

## Mirror insertion

> **Mirror insertion:** this experiment adds per-edit coordinate `m` to a shared low-rank edit basis so that multiple logical edits can be represented without storing each full edit matrix.

- Native method: independent projected full edits.
- Insertion point: shared rank-4 edit basis coefficients and optional private rank-q factors.
- The coordinate is persistent; it is stored with the edit bank.
- The direct control is ordinary PCA plus the same private residual allocation.

## Prior-art delta and controls

PA140/PA135 motivate shared edit bases and locality constraints. The experiment measures the explicit shared/private storage frontier and compares a native PCA-plus-residual encoding with identical functions.

Controls: projected full edit bank; shared PCA only; shared PCA plus private residuals; exact native PCA/shared-private serialization.

## Result

The synthetic 32x32 task used 24 training edits and 8 held-out edits over two development seeds. At zero private variation, rank zero reconstructed edits nearly exactly at about 17.4 KB versus 123.6 KB projected full. At heterogeneity 0.05, rank 2 reduced NRMSE to 0.0016–0.0046 at about 33.2 KB; at 0.2, rank 4 reached 0.011–0.018 at about 47 KB. Protected-key leakage stayed below 6e-8. The native PCA/shared-private control matched outputs and MAC proxy exactly. Full basis reconstruction also cost more MACs than projected full application in this accounting.

## H / T / D / C / U

- **H:** compact shared edit state with private residual allocation can retain held-out function and locality.
- **T:** synthetic 32x32 linear edits; 8 protected keys; rank 4 shared basis; private ranks 0/1/2/4; heterogeneity 0/0.05/0.2; two development seeds.
- **D:** FAIL for Mirror-specific attribution. Native PCA plus identical private factors exactly explains the observed frontier. Fresh remained sealed after this exact alias.
- **C:** ordinary PCA plus per-edit low-rank residual allocation is sufficient.
- **U:** no learned model editor, semantic edit success, adaptive allocator or hardware serving latency was tested.

Facts and limits are recorded in [STATUS.md](STATUS.md), [PROTOCOL.json](PROTOCOL.json) and [RESULTS_CORE.csv](RESULTS_CORE.csv).
