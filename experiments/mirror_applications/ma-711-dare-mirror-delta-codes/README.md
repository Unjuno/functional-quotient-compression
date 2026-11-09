# MA-711 — DARE-sparsified task deltas as Mirror codes

Status: **FAIL for Mirror-specific value; development screen only**  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `dfd0adba`

## Hypothesis

H: A shared basis plus small task coordinates can compress DARE-sparsified deltas further while preserving held-out task behavior; gains must exceed native shared-basis/PCA coding, DARE itself, and merge controls.

## Mirror insertion

> **Mirror insertion:** represent task-specific parameter deltas as codes over a shared physical basis after applying DARE at an identical sparsity level.

Controls: independent dense deltas; DARE alone; raw shared basis; DARE+PCA ranks 1/2/4/8; canonical native PCA/shared coefficients; Task Arithmetic mean; TIES-style trim/sign merge. PA175 makes DARE mandatory; PA26/27 require arithmetic and interference controls.

## Results

**FACT:** On dev seed 71101 (D=2048, 16 task vectors, 90% DARE drop), at private fraction ρ=0 DARE sparse state was 20,354B with mean task-output NRMSE 2.878. DARE+PCA rank1 was 17,454B (85.7% of DARE) and NRMSE 1.370; ranks2/4/8 were 25,710/42,222/75,246B and NRMSE 1.476/1.758/2.191. Across ρ=.05/.2 rank1 NRMSE was 1.373/1.412; DARE NRMSE was 2.944/3.100. Raw oracle shared-basis state was 66,814B and NRMSE 0/.051/.198 at ρ=0/.05/.2. Dense deltas were 131,250B. The Mirror and native PCA payload hashes and reconstructed outputs matched exactly at every rank. Mean Task Arithmetic and TIES merge interference NRMSE at ρ=0 was .957 and 1.306, respectively; these are shared-model merge points, not per-task storage equivalents.

**INTERPRETATION:** PCA after DARE reduced bytes by 14.3% at rank1 and reduced this synthetic probe error relative to DARE, but missed the preregistered 25% byte reduction gate and had large absolute task error. Higher ranks cost more bytes than DARE. The exact native PCA alias removes Mirror-specific attribution. The known raw basis is an oracle-aligned upper bound and costs more than DARE here.

**HYPOTHESIS:** A lower-rank shared basis may denoise random DARE masks, but useful compression beyond DARE would require a stronger sparsity/quality frontier and a native-control advantage.

**BOUNDARY:** Synthetic task deltas generated from a known low-rank basis; no pretrained model, actual fine-tuning, natural benchmark, or GPU merge runtime. Only development seed was run because the frozen byte gate was missed; audit remained sealed.

## H/T/D/C/U

- **H:** DARE-sparsified deltas admit an additional shared-coordinate compression point at useful function quality.
- **T:** Development seed 71101; D=2048, 16 tasks, rank8 shared teacher, private fractions 0/.05/.2, 90% DARE drop, PCA ranks1/2/4/8, dense, raw-basis, Task Arithmetic, and TIES controls.
- **D:** **FAIL for Mirror-specific value; development only.** Best PCA point saves 14.3%, misses 25% gate, absolute NRMSE is 1.37, and native PCA exactly aliases.
- **C:** Ordinary PCA/shared-basis coding explains all gains; DARE itself uses fewer bytes than higher PCA ranks.
- **U:** Whether trained model task deltas or natural multi-task benchmarks exhibit a better frontier. No audit opened after the development gate miss.
