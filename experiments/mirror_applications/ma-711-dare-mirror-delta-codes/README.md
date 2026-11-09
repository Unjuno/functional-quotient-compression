# MA-711 — DARE-sparsified task deltas as Mirror codes

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `dfd0adba`

## Hypothesis

H: A shared basis plus small task coordinates can compress DARE-sparsified deltas further while retaining held-out task behavior; any gain must exceed native shared-basis/PCA, DARE itself, and merge controls.

## Mirror insertion

> **Mirror insertion:** represent task-specific parameter deltas as codes over a shared physical basis after applying DARE at an identical sparsity level.

- Controls: independent dense deltas; DARE alone; raw shared basis; DARE+PCA at ranks 1/2/4/8; exact native PCA/shared coefficients; Task Arithmetic; TIES-style trim/sign merge.
- PA175 makes DARE mandatory before claiming incremental compression. PA26/27 require arithmetic and interference controls.

## Protocol

Seeds, delta width, task count, DARE drop, rank sweep, actual serialized payload accounting and decision gates are frozen in [PROTOCOL.json](PROTOCOL.json). Held-out random probes assess each task function. Merge controls are reported as shared-model operating points.

## Results

Pending frozen development run.

## H/T/D/C/U

- **H:** DARE-sparsified deltas may admit additional shared-basis coding at a better quality/byte point.
- **T:** D=2048, 16 tasks, shared rank 8, private fractions 0/.05/.2, DARE drop .9, PCA ranks 1/2/4/8, development seed then three audit seeds.
- **D:** Pending.
- **C:** Ordinary PCA/shared low-rank coding may fully explain the result; DARE alone may dominate after paying index bytes.
- **U:** Natural model fine-tuning, language/task benchmarks, and GPU merge/decode runtime.
