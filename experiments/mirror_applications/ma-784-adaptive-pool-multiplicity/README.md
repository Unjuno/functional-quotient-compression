# MA-784 — Adaptive physical pool size with logical Givens views

Status: **FAIL** for Mirror-specific value and the frozen 25% complete-byte gate.

## Fact

Four logical operators were generated as Givens-conjugated views of one 8×8 expert. At physical K=1, standard pooling uses 1,089 B and has heldout nRMSE .1838; the angle-view version uses 1,309 B, reaches `7.79e-8`, and costs 72 vs 64 MAC/token. At K=2, no-view pool error is .0835 at 1,230 B, while the view reaches `8.05e-8` at 1,450 B. The independent K=4 bank uses 1,477 B and reaches `6.45e-8`. Thus K=1 meets target quality with 11.4% fewer complete bytes than K=4, below the 25% gate; K=2 saves 1.8%. Mirror and direct-angle Givens have identical payload hashes and function values. The K=1 angle-view CPU pass is ~6.15 ms versus ~1.37 ms for the no-view pool.

## Interpretation

The synthetic orbit supports four logical operators from one physical expert, while the no-view K=1/K=2 pools miss quality. However, NPZ compression already stores the four highly related full operators efficiently, so actual byte savings from reducing K are modest. The coordinate operation increases active compute and wall time, and its entire behavior is standard direct-angle Givens/OFT.

## H / T / D / C / U

- **H:** A smaller physical pool plus compact role coordinates can preserve target quality with fewer complete bytes than an independent expert bank.
- **T:** Seed 78401; four roles; fixed angle grid `[-.36,-.12,.12,.36]`; pool K=1/2/4; 256 heldout inputs per role. Compare ordinary pools, independent K=4, Mirror coordinates and native Givens. Actual compressed NPZ, nRMSE, MAC proxy and CPU pass time recorded; no optimizer updates.
- **D:** **FAIL** for Mirror-specific value. K=1 recovers the target at 11.4% fewer bytes than K=4, missing the 25% gate; MAC rises 12.5%, CPU is ~4.5× slower, and native Givens exactly aliases Mirror.
- **C:** The task family was constructed as a one-plane orthogonal orbit, and compression of the full bank reduces the apparent value of shrinking physical pool size.
- **U:** Learned nonlinear experts, real routers, natural task distributions, private residuals, GPU-fused Givens and multi-seed variation.

## Scope

This measures a designed one-plane rotation orbit with a fixed paid role-to-expert table. It does not show that a trained MoE can reduce its pool by 4×. Fresh evidence remained sealed after the exact native Givens alias.
