# MA-503 — Factorized layer × task representation code

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `49e86eb0` (integrated worker evidence through MA-502)  
Prior art: PA96 (ReFT / LoReFT)

## Hypothesis

H: In a shared rank-4 representation intervention space, a per-layer two-angle Mirror View applied to a per-task code can recover held-out layer-task interventions with lower actual bytes than independent pair codes, while matching a generic layer-matrix × task-code control.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m_layer=(theta_1,theta_2)` to a shared rank-4 LoReFT intervention basis so that one task code expresses distinct interventions at different layers without storing a separate intervention for each layer-task pair.

- Native method: frozen representation model with learned LoReFT-style intervention coefficients.
- Shared object: one known synthetic orthonormal 64×4 intervention basis (its bytes are charged in every payload).
- View: two Givens angles per layer, rotating the 4D task code in two fixed coordinate planes.
- Logical objects: 8 layers × 32 tasks; evaluation holds out layer-task pairs while every layer and task remains observed in training.
- Simpler controls: shared task code reused at every layer; generic full 4×4 layer matrix × task code; direct LoReFT coefficient table.

## Protocol (frozen before fresh)

- Dimensions: D=64, rank=4, L=8 layers, T=32 task codes.
- Fresh worlds: 50320, 50321, 50322; seeds 0,1,2. Each world/seed gets a deterministic 20% checkerboard-like pair holdout, with every layer and task represented in training.
- Development worlds: 50300, 50301; seeds 0,1,2. Development may select only optimizer steps from {400,800,1200}; the selected common step count is frozen before fresh runs. No fresh metrics may tune settings.
- Two preregistered target regimes: aligned (`rho=0`) and 10% off-orbit coefficient residual (`rho=0.1`). The residual measures where private state begins to be needed.
- Data generator: coefficients are `c[l,t]=R(theta_l) z[t] + rho*e[l,t]`; hidden intervention is `B c[l,t]`, with fixed shared B. The two independent Givens planes are (0,1) and (2,3).
- Fit Mirror angles/task codes and generic controls only on observed training pairs. Test on held-out pairs. No target coefficients from held-out pairs may enter fitting.

## Controls and metrics

1. Independent full D-dimensional intervention table (upper bound).
2. Direct LoReFT basis plus one rank-4 coefficient per layer-task pair (storage control / upper bound).
3. One shared LoReFT task code reused across layers (ordinary tying).
4. Generic full layer matrix `A_l` × task code `q_t` (strong native factorized control).
5. Mirror Givens layer View × task code (candidate).

Report held-out and observed-pair NRMSE, exact serialized inference payload bytes, bytes/task, transform/application MAC proxy, optimizer steps, fitting wall time, and decode wall time. All tensors and metadata in the serialized payload are charged. The common fixed basis is serialized in every method.

## Decision gates

- PASS for the registered claim only if Mirror meets held-out NRMSE ≤0.05 in all fresh worlds at `rho=0`, uses ≤80% of the direct LoReFT coefficient-table bytes, and beats both shared-task tying and generic full-matrix factorization on a declared quality/byte Pareto point.
- PROMISING if it passes quality/storage against the table and tying but the generic factorization matches it or runtime cost is unfavorable; this is not a Mirror-specific win.
- FAIL if held-out quality or byte gate fails, or a simpler control matches quality at equal/fewer actual bytes.
- `rho=0.1` is a boundary map, not a required pass; report whether private coefficients restore quality and their full bytes.

## Boundaries

Synthetic frozen representation vectors only. The shared basis is supplied by construction; this is not a pretrained-model LoReFT reproduction, downstream-language test, capacity proof, or claim that the number of combinations is independent capacity. Any oracle target/basis use is disclosed and charged where serialized.
