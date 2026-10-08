# MAT01 / MAT02 — CPU adapter-bank Mirror falsification and private-residual follow-up

**Research-only observational report (2026-10-09).** Two independently preregistered CPU experiment series on sklearn's 8x8 handwritten digit images; they are NOT a reproduction of the LoGo method nor a real LLM. The full original MA-1178 stays **UNTESTED**. We ran the negative results rather than claiming useful compression.

## Frozen order, identity and evidence

- MAT01 protocol GitHub commit `5d4c37a1ba70b162fa09bbe2182a4e2d206f5eac`: dev [11,12,13], fresh [101..105].
- MAT02 pre-fresh amendment commit `c7b7b517c66d757930baf4b7054535c13a0dcf2a`: dev [21,22,23], fresh [201..205]. It was created **after** MAT01's negative result. No retroactive alteration of MAT01.
- Dataset sklearn `load_digits`, 1,797 images, original IDs stratified train 75%/audit 25%; normalization learned from training images only. All target conditions use the **same 300 original support image IDs** within a seed and label set, with 4 role changes: diagonal shifts, Gaussian additive noise, pixel dropout. Six independent source LoRAs train on horizontal/vertical shift, blur and contrast source roles.
- One source-trained 64-hidden GELU base classifier; after freezing, source role-specific rank-4 LoRAs are trained. Derive source-only U[64,4], V[4,10], mean core C[4,4] and source PCA core dictionary (4 atoms of 4x4). For unknown target transformations, train 4D Mirror angles, 4D plain linear coefficients, native independent rank4/rank2, full 16D core and/or rank1/rank2 private deltas on the same target support and update budget.
- All task losses are **heldout fresh per-original-image classification cross entropy (nat/example)** and accuracy. Actual named NPZ model bytes include base, preprocessing, paid task-bank representation and metadata. The heldout dataset ID set remains unseen by every optimiser.
- Fixed CPU PyTorch 2.10.0+cpu, sklearn 1.8, NumPy 2.3.5, torch+BLAS threads1, B128, 20 warmups /100 calls per method (virtual EPYC CPU, unlocked clock). Native LoGo dynamic router/adapter merging is **not implemented**.

## A. MAT01 — source-aligned low-dimensional codes vs true native low-rank adaptation

**fresh 5 dataset partitions × 4 roles = 120 per-role rows**:

| model | NLL ↓ | Accuracy ↑ | actual total NPZ byte ↓ | CPU P95 ms ↓ |
|---|---:|---:|---:|---:|
| native_lora4 | 0.82696 | 0.82344 | 28595 | 0.18897 |
| fullcore16 | 1.71096 | 0.56244 | 25139 | 0.17193 |
| linear4 | 2.17029 | 0.52656 | 25466 | 0.28084 |
| diag4 | 2.50399 | 0.50844 | 24942 | 0.21690 |
| mirror4 | 2.56999 | 0.50244 | 24967 | 1.07089 |
| base | 3.22497 | 0.48578 | 21867 | 0.06122 |

Native rank4 LoRA wins decisively on task performance. The short Mirror code is weaker than the same-dimensional ordinary source-trained linear code on all five independent seeds; a 16D unrestricted shared-core model still cannot match native LoRA. The source learned 4D left/right subspaces do not capture the transformations needed for heldout target-image shift functions. **MAT01 scientific mechanism gate FAIL.**

## B. MAT02 — source Mirror or linear code plus genuinely private rank2 residual

**Separate fresh 5 dataset partitions × 4 roles = 160 per-role rows**, with independent fresh seeds [201..205]. A follow-up created after MAT01's visible dev/fresh failure; not pooled as one preregistered experiment.

| model | NLL ↓ | Accuracy ↑ | actual total NPZ byte ↓ | CPU P95 ms ↓ |
|---|---:|---:|---:|---:|
| native_lora4 | 0.80122 | 0.82589 | 28563 | 0.13689 |
| linear4_private2 | 0.88895 | 0.75756 | 29785 | 0.39968 |
| mirror4_private2 | 0.90653 | 0.75889 | 29249 | 1.35197 |
| native_lora2 | 0.95610 | 0.74056 | 26195 | 0.14611 |
| fullcore16_private1 | 1.02419 | 0.69944 | 28260 | 0.20807 |
| linear4 | 2.26757 | 0.51778 | 25424 | 0.37263 |
| mirror4 | 2.61236 | 0.49867 | 24888 | 1.15013 |
| base | 3.01332 | 0.48811 | 21835 | 0.05874 |

The rank2 private residual brings Mirror NLL **2.61236 → 0.90653**, demonstrating the private residual carried most needed function variation. But **Mirror+private2 (0.90653) remains inferior to ordinary linear4+private2 (0.88895)** and independent rank4 LoRA (0.80122). In only 1/5 fresh worlds does Mirror+private2 beat its plain linear+private2 counterpart. Paired Mirror minus native NLL by fresh world: 0.14759, 0.07825, 0.16340, 0.00558, 0.13173. Mirror+private2 takes 29249 physical persisted NPZ bytes versus native rank4 LoRA 28563 — **larger, not smaller**. This method is slower in the eager CPU implementation because each Givens core is reconstructed by Python/PyTorch operations.

**MAT02 preregistered mechanism gate FAIL**: quality, persisted bytes, latency, and Mirror-specific non-Mirror paired comparison do not all pass. The logical function isn't regained by free coordinate views; it required paid private capacity. This is not a theorem excluding other Mirror coordinate families.

## Critical truthfulness caveat: no single heavy pass for four different image shifts

Each real task uses **different transformed pixels before the frozen encoder**. Therefore the recorded CPU benchmark (one shared backbone on an identical canonical unshifted input followed by four role readouts) is a **synthetic identical-input readout kernel diagnostic**, not the actual per-role deployment workload. In real multitask evaluation with four differently corrupted input images, the feature encoder must process each distinct image (unless a separately trained canonicalization/shared sufficient statistic exists). Do not claim a real 4-role one-forward speedup from these timings; the quality and role accuracies use separately transformed, correctly heldout inputs, while the runtime fixture uses a common dummy input. Matching native controls were timed on the same dummy input, so relative decoder overhead remains an implementation diagnostic only.

**GPU, true LoGo algorithm, natural language, real multi-output same-image task, and independent dataset-world replication are all NOT DONE.** Despite the feature-input mismatching benchmark caveat, the functional quality negative results on shifted digits remain valid for that limited conditional task.

## Verification and decision

- MAT01 dev: 72 condition-role rows, fresh:120; MAT02 dev:96, fresh:160. Distinct world seed sets; no missing method/role/seed cell. All per-seed source subspace hashes and dataset hashes consistent, all metrics finite.
- MAT01 9 original unit tests and MAT02 5 original tests passed, as did all 17 prior batch unit tests (same-session only). Checkpoint weights were not stored; deterministic seeded retrain from original immutable source is the replay route.
- MAT01 scientific pilot **FAIL** (no Mirror-specific beneficial Pareto), MAT02 **FAIL** (private improves but simple native/linear remains better). Parent MA-1178 **UNTESTED** until real LoGo and natural task quality/performance are reproduced.
- Confounds: source rank4; extreme unseen diagonal input shifts; fixed 320/145/175 optimizer steps; role-specific target train data; actual serializer metadata; unloaded vs resident task buffers; virtual CPU kernel overhead; five correlated partitions from the same 1,797 images.
- Useful next experiment: compare source rank R=8/16 vs R=4, source/target task similarity, and compact learned canonicalizing input map before the shared trunk. Freeze a new distinct fresh set before running. Against any proposed Mirror win, ordinary 4D linear/diagonal codes and independent LoRA remain compulsory controls.

## Source/replay

- [MAT01 frozen protocol](MAT01_FROZEN_PROTOCOL.json), [MAT02 independent frozen amendment](MAT02_FROZEN_PROTOCOL.json)
- [MAT01 source](source/run_mat01.py), [MAT01 unit tests](source/test_mat01.py), [MAT02 source](source/run_mat02.py), [MAT02 unit tests](source/test_mat02.py)
- [Complete per-world raw CSVs](results/) and [compact core](results/RESULTS_CORE.csv), [verification](results/VERIFICATION.json).
- All changes are restricted to isolated research branch. Main and worker-ready MA-255 queue remain untouched. No automatic status/claim modification.
