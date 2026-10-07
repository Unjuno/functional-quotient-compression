> **SUPERSEDED — historical architecture proposal (2026-10-06).**  
> The current design no longer Mirrorizes the whole/shared computation path by default.  
> Start with [CURRENT_STATE_2026-10-07.md](CURRENT_STATE_2026-10-07.md) and [ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md](ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md).  
> This file is preserved as provenance for the router-free / deterministic-view stage of the research.

# Mirror Transformer — current architecture proposal

Date: 2026-10-06  
Status: **active architecture hypothesis; not yet a demonstrated capacity improvement**

This document consolidates the current architecture direction after the MN, MT, router-free RF, and reachability-adjusted analysis lines. Historical router-based experiments remain valid records; this document describes the **current proposed next architecture**, which is router-free by default.

## 1. Goal

The target is not to create independent experts and then route among them. The target is to train **one shared Transformer parameter set** so that useful rules remain accessible under multiple fixed observation/computation coordinate systems.

The intended capacity mechanism is:

> store common rules once in shared weights, expose them through multiple low-description Mirror views, and spend the saved representational budget on additional rules instead of duplicated common structure.

Deterministic Mirror views do **not** create new Shannon information. A capacity claim requires a measured rate-quality advantage at fixed serialized bytes and controlled training/inference work.

## 2. Core architecture

The default model is a decoder-style shared Transformer with no learned per-input router:

```text
observation / token stream
    -> optional small source encoder
    -> deterministic Mirror coordinate
    -> one shared Transformer theta
    -> task head / decoder
```

The same learned parameter set `theta` is used for every Mirror view.

### 2.1 Mirror address

A Mirror configuration may depend on known external coordinates:

```math
m = M(s, t \bmod P_T, \ell \bmod P_L, g, v)
```

where:

- `s`: sensor or modality identifier;
- `t`: logical token / time index;
- `P_T`: token-period length;
- `ell`: Transformer layer index;
- `P_L`: layer-period length;
- `g`: optional externally specified global state;
- `v`: optional viewpoint/resource/configuration identifier.

No learned router is required to infer these identifiers in the base design. They are known coordinates of the observation process.

### 2.2 FFN Mirror operator

For an FFN intermediate vector `v = W_up u + b_up`, one general invertible Mirror family is

```math
Q_m = \exp(A_m)
```

and

```math
F_\ell(u;m)
=
W_{down,\ell} Q_m^{-1}
\phi\!\left(Q_m(W_{up,\ell}u+b_{up,\ell})\right)
+b_{down,\ell}.
```

A single fixed invertible `Q` can be folded into ordinary FFN weights; therefore the research value comes from **requiring one shared weight set to work across several distinct views**, not from claiming that one fixed coordinate change expands the function class.

## 3. Mirror family

Rotation-only Mirror is no longer the preferred family.

The candidate generator space should include:

- skew-symmetric components: rotations;
- symmetric-traceless components: anisotropic stretch / squeeze;
- mixed non-normal components: shear-like transformations;
- structured low-description approximations: block, sparse, low-rank, or decoder-known bases.

Global scalar scale is a control, not assumed useful.

Dense data-derived `d x d` generators are not free: if they must be stored, their serialized bytes count against the model. The preferred implementation should therefore seek **high-utility, low-description structured generators**.

## 4. Mirror selection

A Mirror direction is not selected merely because it has high gradient energy.

The selection process separates four quantities:

1. **Task utility** — does the direction improve the target behavior?
2. **Baseline reachability cost** — how expensive is the same functional change using ordinary shared-weight updates?
3. **Retention/interference** — what old capabilities are damaged?
4. **Description/execution cost** — bytes, active compute, and training work.

For a reachable local target, the reachability analysis compares

```math
E(a)=
\frac{a^T R_B a}{a^T R_M a},
```

with generalized eigenproblem

```math
R_B v = \lambda R_M v.
```

A large `lambda` means expensive baseline emulation per unit Mirror metric. It **does not** imply task utility or capacity gain. Candidate directions must survive finite-amplitude and finite-training-step validation on separate development data.

See [REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md](REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md).

## 5. Mirror count

Mirror count `K`, Mirror subspace dimension `r`, Mirror strength `rho`, token period `P_T`, and layer period `P_L` are distinct resources.

For zero-mean discrete codes spanning an `r`-dimensional covariance, the finite-support bound is

```math
K \ge r+1.
```

A regular simplex reaches this support lower bound. This is **not** a theorem that `K=r+1` maximizes capacity, generalization, or training speed.

The optimization target is therefore:

> choose the smallest K whose additional view improves the measured capability/retention frontier after counting its storage and compute cost.

## 6. Token-period Mirror

A token-period schedule uses deterministic phases such as

```text
1, 2, 3, 1, 2, 3, ...
```

but each phase may be a genuinely different transformation, not merely a different angle of the same rotation.

The intended role is temporal multiplexing: one shared Transformer learns to operate correctly through several recurring internal coordinate systems.

Important constraints:

- phase is tied to logical token offset, not chunk boundaries;
- KV-cache continuation must preserve phase;
- padding/BOS/document-reset rules must be explicit;
- zero-mean codes do not guarantee cancellation because token sensitivities differ;
- task token position must not be accidentally confounded with Mirror phase.

## 7. Sensor Mirror / world-model extension

For learning shared rules of the physical world, each sensor can receive its own fixed Mirror coordinate while sharing the world-model Transformer.

A practical decomposition is

```text
camera -> E_camera -> M_camera --\
lidar  -> E_lidar  -> M_lidar  ----> shared Transformer theta
imu    -> E_imu    -> M_imu    --/
audio  -> E_audio  -> M_audio --/
```

The small source encoders `E_s` normalize different raw sensor spaces into a common latent width. The Mirror then represents sensor/view coordinates inside the shared world-model computation.

### 7.1 First training mode: sensor-parallel same-target learning

For synchronized observations of the same event:

```math
L =
\frac{1}{S}
\sum_s
L_{task}(F_\theta(M_s(E_s(x_s))), y).
```

All sensor branches update the same `theta`. This directly tests whether common world rules can be reused across observation systems.

### 7.2 Later modes

Only after the same-target test is understood:

- cross-sensor prediction;
- sensor dropout / missing sensor;
- held-out sensor calibration;
- viewpoint transfer;
- a fused stream in which sensor tokens interleave.

A sensor-specific Mirror that merely becomes a hidden sensor-ID expert is not evidence of a shared world model. Swap, omission, and transfer audits are required.

## 8. Parallel execution

Distinguish:

- `K`: total Mirror views available;
- `M`: Mirror views evaluated in one training step.

For `M>1`, the logical tensor can be packed as `[B,M,T,D]` and flattened into the batch dimension where valid. Gradients from all views accumulate into the same parameters:

```math
\nabla_\theta L =
\frac1M\sum_{m=1}^{M}\nabla_\theta L_m.
```

Computation before the first Mirror divergence can be shared exactly. After hidden states diverge, later attention/FFN operations cannot be counted as shared compute, although they can still be vectorized.

## 9. Training objective

The base objective is deliberately simple:

```math
J(\theta)
=
\mathbb E_{(x,y)}
\mathbb E_{m\sim P_M}
[L(F_\theta(x;m),y)].
```

The same semantic target is used across views unless a later experiment explicitly studies addressed/private capacity.

Do not add strong output-equality penalties by default. If every view is forced to become functionally identical, the architecture may reduce to a regularizer rather than a capacity mechanism.

## 10. Capacity evaluation

A successful Mirror architecture must do more than improve robustness.

At fixed serialized bytes and controlled training work, increase independently:

- shared-rule load;
- private/independent-rule load;
- number of observation coordinates;
- held-out compositions.

Measure:

- one-view NLL/accuracy;
- worst-view quality;
- held-out composition;
- old-rule retention;
- private-rule retention;
- unseen-view / unseen-sensor transfer;
- actual serialized bytes;
- active MAC/FLOP estimates;
- wall-clock training and inference;
- view-induced hidden-state diversity.

Strong controls:

- Dense;
- Dense + dropout / stochastic regularization;
- repeated same-view exposure;
- random fixed Mirror;
- rotation-only Mirror;
- direct gate;
- matched-byte independent MoE where applicable;
- equal-compute continuation from the same parent checkpoint.

## 11. Current implementation priority

The next model-level work should proceed in this order:

1. reproduce the router-free RF baseline from frozen artifacts;
2. implement structured non-rotation Mirror generators with exact inverse/backprop tests;
3. compare layer-only, token-only, and layer x token schedules;
4. use reachability/task/retention analysis only to propose candidates;
5. require short shared-weight training to validate each candidate before adoption;
6. only then test dynamic K;
7. after synthetic capacity evidence, move to synchronized multi-sensor world-model tasks.

## 12. Evidence boundary

This architecture is a **research proposal synthesized from existing experiments**.

Established so far:

- shared factors can be trained;
- Mirror count and Mirror breadth are distinct;
- packed multi-view execution is numerically feasible;
- current rotation/gain Mirror families have not shown a stable capacity advantage;
- reachability-only direction selection has not shown a stable learning advantage.

Not established:

- an optimal Mirror family;
- an optimal K/rho/period;
- a capacity multiplier;
- a natural-language advantage;
- a multi-sensor world-model advantage.
