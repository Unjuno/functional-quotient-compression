# MA-669 — LoRAHub coefficients from Mirror task code

Test whether a reusable two-dimensional task code can replace full-bank coefficient search when composing eight shared rank-1 LoRA atoms for a new task. The code subspace is learned from disjoint meta tasks; support examples infer each task code. Compare LoRAHub-style gradient-free search, direct native full coefficient fitting, a low-rank native coefficient prior, and independent full matrices.

Measure held-out regression quality, actual complete payload bytes and adaptation compute/evaluations. The native low-rank coefficient control is required because a small task code may simply be ordinary low-rank LoRA composition. This is a synthetic screen, not a LoRAHub benchmark.
