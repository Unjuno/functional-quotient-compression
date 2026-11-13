# MA-919 result — scoped mechanism screen

## H — hypothesis

A shared flow-map MLP addressed by start time and duration may improve held-out map and semigroup composition quality per deployed byte or compute over native start/end conditioning.

## T — what ran

- Base: frozen amendment 01 in `PROTOCOL.json`; analytic 2D linear ODE worlds, three fresh system matrices/seeds 91911/91913/91917.
- Model: same shared 2×64 tanh MLP, Adam, 600 updates, batch 256, CPU single-thread.
- Controls: native `(s,t)` conditioning versus coordinate view `(s, dt=t-s)`. The latter is an invertible reparameterization of the same two native times. Exact matrix exponential is the target oracle.
- Metrics: random held-out endpoint MSE, three-map semigroup composition MSE, actual serialized state-dict bytes, MAC proxy, wall-clock. Two complete deterministic replays had identical map/composition metrics and payload sizes; checkpoint reload output max absolute difference was 0.

## D — PROMISING, narrowly scoped fixed-budget learning-efficiency signal

### Fact

| Metric (mean over 3 fresh systems) | Native `(s,t)` | View `(s,t-s)` | Change |
|---|---:|---:|---:|
| Held-out map MSE | 0.01135 | 0.00587 | −48.3% |
| Composition MSE | 0.00926 | 0.00434 | −53.1% |
| Serialized state dict | 21,585 B | 21,585 B | 0% |
| MLP MAC proxy | 4,480 | 4,480 | 0% |
| Training wall clock, one run | 1.03 s | 0.98 s | −5.5% |

All three fresh systems favored the view on both quality metrics. The coordinate computation adds one scalar subtraction per example, not represented in the MLP MAC proxy. The saved model state contains no coordinate parameters; the mapping is deterministic.

### Interpretation

For this analytic mechanism, start-plus-duration coordinates improved a 600-update learning-efficiency signal at equal model bytes and MLP compute. Because `(s,t) ↔ (s,t-s)` is an ordinary invertible time reparameterization, the result does not show Mirror-specific freedom or increased capacity. A conventional baseline could use the same transformed inputs at no additional state cost. Near-convergence and image-model quality were not tested.

### Hypothesis

The lower error may come from an easier optimization geometry for duration-conditioned maps. It may vanish near convergence or on image flow maps; no PA260 image/FID conclusion follows.

## C — strongest counter-hypothesis

The view only supplies a better-conditioned feature basis for this simple linear-flow task. Native time conditioning can encode the same information, and explicit ordinary time reparameterization can reproduce this result without any Mirror-specific mechanism.

## U — unresolved

Near-convergence behavior; byte-matched conventional Fourier/learned time embeddings; nonlinear/stochastic flow maps; PA260 CIFAR-10/ImageNet-32 models and FID; multi-seed runtime significance; complete inference FLOPs including coordinate arithmetic.
