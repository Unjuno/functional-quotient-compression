# MLP-HYBRID-001 — Is a private output correction sufficient to rescue Mirror?

**Status: CPU mechanism study completed (2026-10-17); preregistered Mirror-specific gate FAIL.** [Usage decision](../../../../docs/phase2/MLP_MIRROR_USAGE_DECISION_2026-10-17.md).

Base: shared gate/up/down SwiGLU; factorized Mirror (two Givens code banks) versus identical-cost native gain. Test optional per-pair bias, shared rank-two output residual, full private output correction, plus baseline native task-specific readout and independent four FFNs.

Primary: 5 new handwritten-digit splits ×2 initial seeds ×2 roles ×11 methods = **220** fits at 550 AdamW updates. Separately preregistered post-primary follow-up: 3 new splits ×2 seeds ×2 roles ×5 methods = **60** fits at 1,400 updates. Validate/checkpoint choose without test optimization. All 560 final/best serialized inference payloads SHA/byte-replayed via separate NumPy FP64. **11 unit tests pass; three complete retrainings reproduce payload hashes.**

At digit attribute classification, mirror+rank2 selected-checkpoint BCE **0.07680** at **16,581 B**, equal-size gain+rank2 **0.08867**; but 5% joint mean/worst-task gate FAIL. mirror+full **0.06904 / 16,773 B**, while the stronger native pair-readout is **0.06884 / 15,573 B**. On noisy digit patch denoising, mirror+full **0.03315 / 24,133 B**, native gain+full **0.03283 / 24,133 B**, native pair readout **0.03286 / 21,589 B**. The native readout is the practical default.

The same digit image corpus is reused across task labels and split worlds; this is not independent generalization to distinct datasets. CPU eager view kernels are slow. No Transformer NLL or real MoE routing benefit proven. Full source and raw model evidence is in the separately downloadable `MLP_MIRROR_USE_20261017_FULL_EVIDENCE.zip`. No change to canonical MA statuses or main.