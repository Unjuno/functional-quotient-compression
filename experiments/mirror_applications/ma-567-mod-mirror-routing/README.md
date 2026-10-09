# MA-567 — Mixture-of-Depths with Mirror compute/function routing

Status: **FAIL for Mirror-specific gate**  
Branch: `research/ma-567-mod-mirror-routing-20261009`  
Base commit: `a84cc98a`  
Prior art: PA111 Mixture-of-Depths

## H

At fixed token routing and compute budget, a small role-conditioned Mirror view on the shared block improves routed-token function quality over vanilla MoD while using fewer bytes than independent role blocks.

## Frozen screen

Synthetic token regression with two role classes, a fixed norm-based top-50% compute router and one shared 8×8 block. The teacher applies role-specific rotations to the shared block for routed tokens; un-routed tokens take the residual path. Compare MoD base block, Mirror role views, direct two-basis coefficients and independent role blocks. Two development seeds, 512 train/512 test tokens. Routing masks are identical across methods. Report output nMSE, actual serialized payload bytes, router/block/view MAC proxy and isolated wall time.

PASS requires routed-token nMSE ≤1e-4, ≥20% fewer bytes than independent role blocks and ≥10% fewer than direct coefficients at equal quality. FAIL if direct controls match or routing/quality gates fail. Fresh sealed on exact alias.

## H / T / D / C / U

- **H:** Mirror adds function choice without changing MoD's compute allocation.
- **T:** Fixed top-50% norm router, two token roles, 512 train/512 test, two seeds; eight rows.
- **D:** FAIL. Mirror/direct role views match exactly at 1204B and nMSE 0; independent blocks use 1235B (only 2.5% savings). Vanilla MoD uses 959B but nMSE .24655.
- **C:** Direct two-basis role coefficients and vanilla MoD.
- **U:** Learned router, Transformer attention/MLP execution, language-model NLL and GPU dispatch latency.
