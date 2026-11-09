# MA-788 — Utility-weighted physical expert copies vs logical Views

Status: **FAIL** for Mirror-specific value; utility-ranked hybrid allocation is promising for follow-up with native controls.

## Fact

Sixteen child functions share a parent plus one rank-1 direction; the two highest-utility children also contain private residuals. All-view rank-1 codes use 2,295 B but have utility-weighted nRMSE .339. Copying the top two utility children yields .0409 at 3,828 B, 58.7% fewer bytes than the all-copy bank (9,266 B, zero error) and far better weighted error than random K=2 copies (.339 at 3,834 B). The utility hybrid's utility-weighted MAC proxy is 255 vs 144 for all copies and 288 for all Views. The all-View and native rank-1 LoRA payload hashes and metrics are identical. Zero optimizer updates; child code fitting uses checkpoint-derived projections in this screen.

## Interpretation

The synthetic utility score correctly identifies the two children whose private variation makes a shared code fail. Paying full weights for those roles while keeping compact codes for the rest creates a useful quality/storage point. Random allocation at the same K fails to recover high-weight tasks. This is evidence for utility-aware physical/private allocation, not Mirror-specific behavior: the View portion is exactly native rank-1 LoRA.

## H / T / D / C / U

- **H:** High-utility children should receive physical copies when Views cannot fit them; aligned low-utility children can use compact codes.
- **T:** Seed 78801; 16 synthetic 12×12 children; one shared residual direction; private residuals on the two highest-utility roles; compare all View, all copies, utility K=2/4/8 hybrids and random K=2; actual NPZ bytes, utility-weighted error and MAC proxy recorded.
- **D:** **FAIL** for Mirror-specific value. Utility K=2 is a useful measured hybrid point, but its shared code exactly aliases native LoRA; all-view quality misses badly, and code application adds compute.
- **C:** The benefit comes from the preassigned utility ranking locating private outliers and a standard low-rank code serving aligned roles.
- **U:** Learned utility from routing statistics, trained MoE continued pretraining, multiple seeds and realistic router/task quality.

## Scope

Synthetic operators and known utility scores only. The hybrid uses a simple low-rank code identical to LoRA; no Mirror advantage, capacity claim, or measured GPU-hours saving is asserted.
