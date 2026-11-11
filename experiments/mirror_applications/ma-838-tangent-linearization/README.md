# MA-838 — Tangent coordinates for nonlinear fine-tuning

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / LINEARIZATION  
Base commit: `9ddefe35`  
Prior art: PA226 (linearization of fine-tuning) and PA33 (intrinsic-dimensional fine-tuning).

## Hypothesis

H: A shared low-rank Mirror basis learned from task tangent updates preserves nonlinear heldout quality with >=25% fewer bytes than independent updates, outperforms random intrinsic-dimension projection, and strictly beats native PCA/tangent factorization.

## Controls and audit

A small tanh MLP is perturbed at two preregistered scales. We fit the local tangent update from support data, learn a rank-4 shared basis from six tasks, and adapt four heldout tasks in that basis. Exact nonlinear teacher updates, independent support-fitted updates, random subspace, native PCA and Mirror are compared. We directly measure the residual between the linearized and nonlinear model.

This is a synthetic MLP screen, not pretrained-model or language-model evidence. See [PROTOCOL.json](PROTOCOL.json) for gates and accounting.
