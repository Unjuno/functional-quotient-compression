# MA-427 — Task-conditioned Deep Equilibrium maps

## H — hypothesis

A shared contractive DEQ map plus a compact rank-1 task coordinate may realize multiple stable fixed-point functions with lower serialized state than full per-task task-bias maps. A Mirror-specific claim also requires an advantage over the matched native rank-1 bias conditioner.

## T — execution

Synthetic 4D contractive `tanh` DEQ. One development world (42701) fixed the rank-1 task-bias direction from four development task vectors. Fresh worlds 42711–42713 each contained eight aligned tasks and four out-of-span boundary tasks; 128 support and 64 heldout inputs/task. Compared shared/no-code, rank-1 Mirror, the byte-identical native rank-1 bias control, and per-task full bias. Each task code was inferred in closed form from support fixed points. No optimizer updates; 1,536 support examples/world. Fixed-point iteration, residual and contraction bound were recorded. Actual canonical NPZ inference payload bytes were measured.

## D — FAIL

Aligned Mirror mean heldout MSE was `4.42e-18`, max fixed-point residual stayed below `3e-11`, and mean iterations were 18.79, matching the full-bias upper control's MSE at numerical precision. The shared/no-code MSE was 0.00802. However, the same rank-1 task-bias conditioner had exactly identical arrays, serialized bytes (1,450 B), output MSE and iteration counts. The independent full-bias bank serialized to 1,064 B; thus Mirror used 36% more bytes and missed the <=60% storage gate. Boundary task Mirror MSE was 0.01233 versus near zero for full bias. The preregistered Mirror-specific Pareto gate failed in all fresh worlds.

## C — strongest counter-hypothesis

The result is ordinary low-rank task bias conditioning of a contractive map. It does not require Mirror-specific structure; the task code is simply a coefficient for a shared bias direction. At this tiny dimension, serialization container overhead also erases the theoretical raw-float savings.

## U — unconfirmed

No language model, learned DEQ, nonlinear solver, adaptive stopping, GPU throughput, or natural task family was tested. Larger dimensions and a packed deployment format could alter storage, but are new experiments.

## Fact / Interpretation / Hypothesis

- **Fact:** aligned quality and stability passed; Mirror and native rank-1 bias control have identical function and serialized state. Mirror payload 1,450 B; full-bias bank 1,064 B.
- **Interpretation:** low-dimensional coordinates can parameterize multiple equilibria, but this experiment found no Mirror-specific advantage and no storage win at d=4.
- **Hypothesis:** larger task banks with truly low-rank task variation might benefit if actual packed bytes beat native conditioning and solver cost stays stable.
