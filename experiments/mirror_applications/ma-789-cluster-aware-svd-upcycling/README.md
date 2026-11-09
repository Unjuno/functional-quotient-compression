# MA-789 — Cluster-aware SVD upcycling with shared View basis

Status: **FAIL** for Mirror-specific value.

## Fact

Twelve synthetic 12×12 experts are assigned to three semantic activation clusters. At private noise σ=0, a global rank-3 basis plus cluster means reaches heldout-member nRMSE `7.94e-8` with 4,938 B vs 7,510 B independent (34.2% fewer). Router accuracy is 1.0. Three separate cluster-specific rank-3 SVD bases use 8,162 B and have heldout error .0515. Mirror and native global PCA have identical payload hashes and metrics. At σ=.05, global-basis error rises to .316 and the per-cluster SVD error is .333; independent weights remain exact.

## Interpretation

A shared subspace can be more compact and accurate than three separately stored truncated cluster bases when synthetic clusters share the same residual directions. Modest private heterogeneity destroys heldout recovery. The gain is ordinary global PCA/SVD and does not establish Mirror-specific value or trained cluster-aware upcycling quality.

## H / T / D / C / U

- **H:** A shared cross-cluster residual basis plus cluster/member codes preserves cluster-aware expert behavior at lower complete bytes than separate cluster-specific SVD subspaces and full experts.
- **T:** Seed 78901; 3 clusters × 4 experts; dimension 12; global rank 3; private noise 0/.05; last expert in each cluster held out from basis fitting; synthetic nearest-centroid router fit on 64 activations/cluster. Independent, cluster-specific SVD, global PCA/Mirror and routing controls. Actual compressed NPZ bytes, heldout operator error, router accuracy and reconstruction MAC recorded; zero optimizer updates.
- **D:** **FAIL** for Mirror-specific value. The aligned global code saves 34.2% vs full experts and beats per-cluster SVD in this world, but native global PCA exactly aliases it; σ=.05 misses quality badly.
- **C:** The generated experts share one global rank-3 residual span; standard PCA captures it without Mirror-specific geometry.
- **U:** Trained Cluster-Aware Upcycling, semantic clusters from natural activations, self-distillation, routed task utility and private-state allocation.

## Scope

Synthetic linear experts and a synthetic nearest-centroid router only. This does not reproduce the PA207 training pipeline or establish downstream task quality. Fresh identities remained sealed after native PCA matched exactly.
