# Gauge-aware natural-LoRA orbit audit (research intake)

Date: 2026-10-08 JST  
Scope: **pre-screen MA-1096/1097/1098/1099/1114/1115**. This is NOT an MA experiment completion or a Mirror architecture implementation. Existing scientific statuses are unchanged.

## Research question

Are **independently trained, natural** task-specific LoRA updates expressible by one shared pair of low-dimensional directions plus a genuinely small task/role code `m`? A cheap-code mechanism only becomes useful when it outperforms the strongest native alternatives at **actual downstream task quality, serialized bytes and runtime**. Earlier deliberately aligned synthetic teachers do not settle this question.

Relevant direct prior art:
- PA351 — [Compress then Serve](https://proceedings.mlr.press/v267/gabrielsson25a.html): shared bases + per-LoRA scaling, clusters and realistic 1000-adapter serving;
- PA352 — [Compress then Merge](https://proceedings.mlr.press/v306/he26h.html): common left/right LoRA spaces and small cores, but solves **one merged adapter**, not many independently selectable functions;
- PA353 — [EigenLoRAx](https://arxiv.org/abs/2502.04700): principal subspace from existing trained adapters for new-task adaptation;
- PA356 — [Reusable pretrained spectral basis](https://arxiv.org/abs/2605.07302): leading **base-weight** singular vectors remain stable under fine-tuning; this does **not** imply that fine-tuning delta subspaces coincide;
- PA357 — [Crowded in B-Space](https://arxiv.org/abs/2604.16826): apparent output-side LoRA B sharing/interference;
- PA358–360/367 — GLoRA, GL-equivariant Learning on LoRAs, W2T, LoRA-RITE: nonidentifiability, canonicalization and invariant optimization;
- PA364/365 — LRAgent and PReCache: already share LoRA-agent KV base plus compact specialist state.

## Critical gauge invariant

With LoRA delta `D = B @ A`, and **any** invertible rank-coordinate matrix `G`:

```text
B' = B @ G
A' = solve(G, A)
D' = B' @ A' == D
```

The stored factor coordinates `A,B` are not uniquely identifiable. An honest subspace test uses the singular values and orthogonal **row/column projectors of D**, not raw elementwise B or A similarity. Nonorthogonal G reparameterizations MUST leave the audit score and model predictions unchanged (allowing floating-point tolerance). Individual singular vectors in a degenerate eigenspace also have arbitrary rotations; compare projectors.

The included `source/orbit_audit.py` computes compact canonical SVD from QR factors, and shared input/output bases by fit on **training tasks only**:

```text
U, V = shared_basis({D_train})
C_t = U.T @ D_t @ V
D_hat_t = U @ C_t @ V.T
D_hat_t_diag = U @ diag(diag(C_t)) @ V.T
```

The dense core `C_t` is a **native control comparable to CtS/CtM**, not a Mirror novelty. The diagonal or sparse core is a preliminary low-description-code *screen*, not proof that Givens/View codes train or win. The script deliberately computes **oracle reconstruction of held-out task weight updates**, which is easier than learning a new-task code from training examples.

## Running the mathematical intake

From repository root, with numpy and pytest installed:

```bash
python -m pytest -q experiments/mirror_applications/research_intake/natural_lora_orbit_20261008/tests
```

The four tests cover: nonorthogonal GL(r) gauge reparameterization of fixed LoRA deltas, related-versus-independent oracle train/audit projection, subspace edge cases, and refusing invalid model revision or task/split metadata. They were run in a CPU container with numpy 2.3.5 and passed 4/4; these are **harness tests only**.

To screen a real adapter family, first export its matched-layer LoRA factors as local `.npz` files holding numeric arrays `B` with shape `[d_out,r]` and `A` with shape `[r,d_in]`. Do **not** include weights from unrelated base revisions. Use exactly matched model **revision/hash**, module path, tensor orientation and adapter scaling; convert adapters with the same output delta convention before comparison.

Create a manifest (all paths relative to its location), for example:

```json
{
  "base_revision": "EXACT_BASE_COMMIT_SHA",
  "layer": "model.layers.0.self_attn.q_proj",
  "items": [
    {"task": "dev_task_1", "split": "train", "file": "dev_task_1.npz", "base_revision": "EXACT_BASE_COMMIT_SHA"},
    {"task": "audit_task_1", "split": "audit", "file": "audit_task_1.npz", "base_revision": "EXACT_BASE_COMMIT_SHA"}
  ]
}
```

```bash
python experiments/mirror_applications/research_intake/natural_lora_orbit_20261008/source/orbit_audit.py \
  --manifest /path/to/manifest.json --rank 4 --sparse-q 4
```

Do not use natural task checkpoint weights in the **train** subspace fit if the same task is designated **audit**. The manifest labels must be reconciled with the authoritative original model/adapters; a string revision is a provenance check supplied by the worker, not external cryptographic attestation.

## Required next experimental steps before scientific adoption

1. Select at least two related and two unrelated natural task adapter families **sharing an identical verified pretrained base**. Freeze train/audit task identities, source model revisions, example splits and baseline code before opening audit tasks.
2. Extract canonical delta SVD without building dense tensors where practical; report per-layer row- and column-projector angles, spectrum/rank, distinct task diversity and randomized gauge-invariance perturbations.
3. Compare trained standalone LoRA; a pooled learned adapter shared basis; CtS with native clustering; CtM (single merged result for composition only); EigenLoRAx; VB-LoRA; MetaTT; pretrained-weight spectral coefficient codes; a plain diagonal/FiLM/rank-1 code; structured Mirror Givens/Householder code; shared Mirror with private low-rank residual; an independent model upper reference.
4. Report **the full natural orbit/private frontier** at fixed task score, varying physical basis rank `k`, per-task code bits, useful task count `K`, residual rank, and task heterogeneity. If `m` cannot beat a k-by-k dense core or cheap native controls, report FAIL/no Mirror-specific gain. A per-task `k×k` matrix should not be called an ultratiny code.
5. **Do not confuse storage accounting with this script's error output.** The minimum raw-float shared-core storage is `k*(d_out+d_in) + K*k*k` values before scale, metadata, serialization, task lookup and any private residual. Native independent LoRA factors require `K*r*(d_out+d_in)` values before overhead. Neither formula establishes actual serialized bytes or execution time; implement a real serialization/decoder and compare wall time.
6. Fit `m` from permitted task data without the oracle held-out LoRA delta. Evaluate downstream NLL/accuracy/retention and calibrated adaptation compute, not only delta Frobenius reconstruction. For a new adapter bank, basis may need to be relearned; account for that cost.
7. If the adapter is used in attention, compare MA-691 exact same-state cache algebra with **LRAgent/PReCache native low-rank cache reuse**. Probe cache provenance, real pointer aliasing, context length, prefill, TTFT, latency and memory, not just numerical cache agreement.

## Interpretation

A positive representation screen supports only: `a held-out learned task update can be approximated by a shared coordinate basis at a specified oracle error`. It does **not** prove task accuracy, adaptive learning feasibility, parameter novelty, independent information creation, cache sharing, GPU speedup or real-world compression until those follow-up tests succeed.

No source model weights, user data or private adapter material are committed in this intake. New MA1096..1115 remain UNTESTED.
