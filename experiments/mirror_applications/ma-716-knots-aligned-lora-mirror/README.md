# MA-716 — KnOTS aligned LoRA Mirror bank

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / QUALITY  
Base commit: `d65f326e`  
Prior art: PA180 KnOTS; PA04 LoRA; PA27 TIES; PA02 DARE.

## Hypothesis

H: After a shared orthogonal KnOTS-style alignment, an aligned sparse task-coordinate code can preserve a bank of heterogeneous low-rank updates at lower complete serialized bytes than aligned dense cores and native low-rank adapters, beyond gauge-only effects.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, an aligned sparse coordinate code over a joint-SVD task update basis, so that multiple logical LoRA updates can be represented without storing each full factor pair independently.

This is a synthetic mechanism screen, not an empirical LoRA-bank claim. The shared basis and task codes must be paid. Native aligned dense core, independent factors, and TIES/DARE-style sparse deltas are mandatory controls. Gauge transforms preserve the functional delta and cannot count as new multiplicity.

## Frozen protocol

D=128, rank r=8, 16 task updates, one shared orthogonal output/input basis fit on eight development tasks only. Eight held-out tasks are generated with aligned plus controlled off-subspace components; no audit tasks in this screen. Compare independent rank-8 LoRA factor serialization, native dense aligned cores, diagonal aligned cores, sparse top-k aligned cores with k in {2,4,8,16,32}, and shared basis plus private rank-1 residual. Refactor factor pairs by three deterministic invertible non-orthogonal GL(r) matrices and verify predictions/projector metrics remain invariant. Development seed 71601 fixes the code/pruning selection; no fresh/audit set opened in this mechanism screen.

### Gates

A useful screen requires held-out update NRMSE <=0.05, at least 25% total actual-payload savings versus both independent LoRA and native dense aligned cores, and no exact alias to a simpler diagonal/sparse native code. Any NRMSE >0.05, complete-payload miss, or native alias is FAIL for Mirror-specific value; invalid gauge invariance/replay is NOT ESTABLISHED.

## Storage and compute

Serialize the single complete bank and include basis, per-task codes, private residuals, indices, scales, metadata and independent factors. Report projection/multiply proxy and wall time. No neural task, pretrained base, training or serving result is claimed.

## Results — development seed 71601

**FACT:** Independent rank-8 LoRA payload was 2,260,670 B (held-out NRMSE .290); native dense aligned-core payload 2,033,159 B (NRMSE .313); shared sparse top-2 Mirror/native payload 2,027,007 B (NRMSE 1.070); native diagonal core 2,027,342 B (NRMSE .956). Mirror sparse and native sparse payload hashes and outputs are identical. Gauge refactorization changed independent-factor predictions by at most 3.4e-16.

**INTERPRETATION:** Every held-out reconstruction misses the frozen .05 quality threshold. The modest byte savings versus independent LoRA comes with severe distortion; Mirror does not improve on native sparse aligned coefficients. This result is a synthetic mechanism screen only. The implementation is a pooled-SVD alignment analogue, not a reproduction of the full KnOTS pipeline; the frozen protocol's top-k and multiple gauge perturbation grid was not fully swept because the initial point already missed quality by a wide margin. That deviation is disclosed and no audit data were opened.

**HYPOTHESIS:** A basis learned from real related adapters or private residual allocation could move the frontier, but this run provides no evidence for that possibility.

**D:** FAIL for this frozen synthetic screen; no audit tasks opened.

**C:** The generated deltas have unrelated residual noise, and the simplified pooled-SVD alignment is not the full KnOTS algorithm; real adapter families or exact alignment may behave differently.

**U:** Natural task accuracy, actual pretrained base/adapters, exact KnOTS reproduction, TIES/DARE quality, fresh worlds and serving runtime.

### H / T / D / C / U

- **H:** Aligned sparse coordinates preserve held-out LoRA functions while reducing complete bytes beyond native controls.
- **T:** 16 synthetic D=128 rank-8 updates; eight fit tasks and eight held-out whole tasks; dev seed 71601; independent LoRA, dense aligned, diagonal and sparse controls. Gauge refactorization prediction delta <=3.4e-16.
- **D:** FAIL; best NRMSE .290 independent / .313 aligned dense, Mirror sparse NRMSE 1.070 and exact native sparse alias.
- **C:** Synthetic task structure and simplified alignment may not match empirical adapters or KnOTS.
- **U:** All natural-task and production-serving claims.
