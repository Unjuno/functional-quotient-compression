# MA-670 — Compress source LoRA bank before composition

Measure whether many source LoRA modules can be represented by shared rank-1 atoms and compact source/target codes before composing target adapters. Compare LoRAHub's complete source factors plus dense composition coefficients with a Mirror-coded shared bank, an identical native VeRA-style shared-scaling control, and full target matrices.

Target coefficients are supplied by the synthetic task construction to isolate bank compression from coefficient search, which MA-669 tested. For nonzero private variation, residual source factors and target coefficients are charged. This is a synthetic mechanism screen, not a language-model benchmark.
