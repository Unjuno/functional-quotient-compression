# MA-840 — Shared tangent basis with private nonlinear residual allocation

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / PRIVATE RESIDUAL  
Base commit: `99527bed`  
Prior art: PA226 (linearized fine-tuning) and PA18 (VeRA).

## Hypothesis

H: A shared rank-4 tangent basis plus the smallest support-derived private residual rank preserves nonlinear heldout quality with >=25% fewer bytes than independent updates and beats native PCA/residual and VeRA-style controls.

## Controls and boundaries

A synthetic tanh MLP has six ordinary tasks and four heldout tasks, two with explicit private tangent directions. We measure residual ranks 0/1/2/4 at two delta scales. All bases, codes and private residuals are included in serialized payload bytes. At delta .03, shared-only rank0 meets .02 quality at 3,091 B, while rank4 improves nRMSE .01405 to .00312 but costs 4,783 B vs 4,680 B independent. At delta .20 all tangent variants miss quality. Mirror exactly aliases native PCA/residual. The private basis uses support-derived residuals from heldout task supports. No pretrained-model result. See [STATUS.md](STATUS.md).
