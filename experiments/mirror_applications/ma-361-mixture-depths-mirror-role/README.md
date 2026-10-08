# MA-361 — Mixture-of-Depths with Mirror block-role views

Status: **FAIL at development; fresh sealed.**
Branch: `research/ma-361-mixture-depths-mirror-role-20261008`  
Base: `c935a90`  
Prior art: PA49 Mixture-of-Depths.

## H — Hypothesis

With identical token routing, one shared residual MLP block plus small per-role functional coordinates can preserve native MoD NLL while reducing actual inference bytes and keeping active MACs within 10%.

> **Mirror insertion:** this experiment adds a role coordinate to the shared residual MLP used by routed tokens, to express multiple logical depth roles without separate block weights.

The direct scalar-gate control uses the same coordinates and is the closest non-Mirror alternative.

## T — Protocol and amendment

Synthetic four-class token task, 16 tokens/example, width 16, 3 block roles, top-50% fixed input-derived routing, 1,024 train and 512 test sequences/world, 1,000 AdamW updates and batch 64. Development seeds 36101/36102; fresh seeds 36111/36112/36113 remain locked and unopened. Controls: dense three-block model, native MoD with separate blocks, tied MoD, tied MoD plus direct scalar gates, tied MoD plus Mirror role coordinates.

The original development artifacts did not replay under the current container runtime. Amendment A1 was recorded before any fresh access: preserve the old artifacts as an excluded protocol variant, pin CPU threads to 1, report Python/NumPy/PyTorch versions, and rerun only development seeds without tuning the model or gates. The amended 8 rows now replay exactly.

## D — Development result

**FAIL for Mirror-specific value and the frozen quality margin.** On amended development runs, Mirror used 5,307B and 5,321B versus native MoD 11,053B and 11,084B. All methods routed 8/16 tokens and had the same active-MAC proxy, 786,432,000. But direct scalar gate and Mirror payload/hash/NLL/accuracy were identical in both seeds. Mirror NLL was 0.10016 and 0.11502 versus native MoD 0.09525 and 0.10227. Seed 36102 exceeds the allowed +0.01 NLL margin by 0.00274. Therefore fresh remains sealed.

Measured training wall time was 2.32–2.87s for tied/gated methods versus 2.57–2.65s native in these small CPU runs; this noisy screen establishes no runtime advantage. Training saw 1,024,000 token examples per method.

## C — Counter-hypothesis

The role coordinate is ordinary scalar gating: its exact equivalence to the direct gate control explains the result. Fixed input-derived routing and this synthetic task may not expose a distinct Mirror contribution.

## U — Unconfirmed

Fresh seeds, learned routing, natural language, near-convergence capacity, larger models and optimized inference kernels remain untested. The byte reduction is shared-block tying and does not establish Mirror-specific value.

## Fact / interpretation / hypothesis

- **Fact:** see `RESULTS_CORE.csv`, `VERIFICATION.json`, and the preserved pre-amendment artifacts. Eight amended development rows replay exactly; fresh seeds were not opened.
- **Interpretation:** physical block sharing cuts bytes, while the added Mirror parameter matches a direct scalar gate exactly and misses the NLL margin in one development world.
- **Hypothesis:** learned routing or deeper natural tasks may make depth-role coordinates useful, but that requires a new preregistered experiment.
