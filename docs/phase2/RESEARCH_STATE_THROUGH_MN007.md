# Phase II research state through MN007

Date: 2026-10-03

This document is the canonical research-state summary for Mirror-native work after MN002. It records positive, negative, and corrective results from MN003 through MN007. These experiments are small CPU synthetic studies unless stated otherwise. They do **not** establish a natural-language-model advantage, a general capacity multiplier, or quality-preserving 64x compression.

## Current high-level status

The architectural core remains viable:

- one shared computation can be modulated into multiple input-dependent functional states without storing a full expert matrix per state;
- multiple independent routing heads can be packed into one batched projection rather than executed as a sequential routing chain;
- causal KV-cache reuse remains numerically valid in the tested MLP-centered designs when model weights and past-token states are not retroactively changed;
- shared-state representations can reduce inference description length relative to fully independent embeddings in some controlled settings.

The stronger claims remain unresolved:

- more Mirror states do not automatically yield better quality;
- more routing freedom does not automatically yield better quality;
- learned overlap/averaging is not a universal improvement;
- router collapse did not explain the main failures observed in MN006;
- semantic similarity did not yet self-organize reliably into a shared parameter identity plus a transferable Mirror state;
- additive and low-rank controls remain strong and sometimes better.

## Evidence by experiment

### MN003 — block-parallel routing

Question: if simultaneously computable FFN subspaces are separated and each receives its own routing head, can routing be batched without changing the function?

Result:

- packed and head-loop implementations matched numerically in forward/backward tests;
- on the trained d=32/F=64 CPU case, the 8x2 router itself was 5.27x faster than calling the heads in a Python loop;
- the full FFN was 2.93x faster and one-token model decode was 1.61x faster in that small eager-CPU comparison;
- at d=512/F=2048 and 512 tokens, FFN speedup shrank to about 1.03x because matrix multiplication dominated;
- factorized routing (2x8, 4x4, 8x2) often beat flat 1x16 at lower parameter count, but Dense controls had better median NLL on the held-out composition task.

Interpretation: routing depth can be avoided. Combinatorial routing paths are available, but combinatorial path count is **not** evidence of equivalent independent-expert capacity.

### MN004 — overlapping sparse supports

Question: can several parallel heads choose overlapping Mirror regions and resolve overlap by averaging?

Result:

- learned where-routing and how-routing were both used by the trained models;
- overlap averaging was not consistently better than summation;
- on the learned-support composition task, mean aggregation lost to sum in all 3 paired seeds on OOD NLL;
- dynamic support added measurable CPU cost;
- fixed-support sum and mean have the same function family when correction amplitudes are freely rescalable and no bound/quantizer blocks the transformation.

Interpretation: overlap normalization controls scale; it is not a general conflict-resolution mechanism. Fixed-support mean-vs-sum performance differences must not be interpreted as intrinsic capacity differences.

### MN005 — candidate-region routing

Question: can where-routing be made cheaper by selecting among a small stored bank of sparse regions instead of scoring all coordinates?

Result:

- parameters fell from 35,777 to 27,713 and the complete model file from 146,276 to 122,404 bytes in the lookup configuration (~22.5% and ~16.3% reductions);
- small-model one-token decode did not beat the full-coordinate design and remained slower than Dense;
- in a larger untrained FFN microbenchmark, candidate routing was about 1.14x faster for one token and 3.10x faster for 64 tokens at d=512/F=2048;
- quality was condition-dependent; fixed disjoint routing was a strong control;
- learning an input-dependent overlap strength was not a stable quality improvement.

Interpretation: reducing the search space can reduce metadata/parameter cost and can help runtime when routing work is material, but the restriction can also remove useful routing freedom.

### MN006 — router early-lock-in hypothesis

Question: were the negative dynamic-routing results mainly caused by the router getting trapped by its initial selections?

Result:

- strong early lock-in was not observed in the tested setup: many support selections changed between early and final checkpoints and no routing head monopolized one candidate on >90% of answer positions;
- forcing balanced early usage did not improve the final result;
- a counterfactual candidate-teacher objective did not give stable gains;
- router-only 100-step fine-tuning improved all tested models;
- however, the same extra 100 steps applied to the whole model improved far more: median OOD-NLL improvement was ~92.6%, versus ~24.2% for router-only fine-tuning.

Interpretation: the router can be improved, but the larger remaining issue in this experiment was joint optimization of shared representation, Mirror state, and router, not a simple router-collapse failure.

### MN007 — semantic shared bases and Mirror states

Question: can semantically related concepts be represented as the same parameter identity plus a small state/view code, rather than only as nearby independent embeddings?

Two tests were separated:

1. **Oracle grouping:** family/state structure was provided and the representation mechanism was tested.
2. **Learned grouping:** concept IDs were opaque and grouping had to be inferred from supervised functional behavior.

Oracle-grouping result:

- Additive shared-base + state was strongest near the tested byte budget: 92.09% held-out-composition accuracy at 22,704 B;
- Mirror-Phase reached 88.87% at 22,516 B;
- CP8 reached 83.30% at 19,954 B;
- Mirror-Phase + private residual did not improve the held-out-composition result.

Learned-grouping result:

- Mirror-Phase inference file: 30,789 B versus independent embedding: 50,286 B;
- Mirror-Phase accuracy: 76.71%, NLL 0.8393;
- low-rank-4 was smaller (22,192 B) and had better median accuracy/NLL (77.20%, 0.5267);
- learned Mirror-Phase family ARI was only about 0.062-0.072: above random/null controls but far below the predeclared 0.5 semantic-recovery gate;
- behavior-derived initial groups became less aligned with the true semantic family during learning;
- freezing those initial groups preserved more ARI but worsened NLL in all 3 tested seeds.

Interpretation: a shared-base/state representation can be stored compactly, but task optimization did not automatically recover the intended semantic quotient. Predictive usefulness and human-interpretable semantic grouping can diverge.

## Claims that should NOT be made

Do not claim any of the following from MN001-MN007:

- that Mirror-native models generally outperform Dense models;
- that K^R routing combinations imply K^R independent-expert capacity;
- that overlap averaging resolves gradient conflict;
- that router collapse was the main cause of the observed failures;
- that semantic similarity has already been compressed into shared parameter identity;
- that Mirror transforms beat generic factorization;
- that CPU microbenchmarks predict GPU or production LLM throughput;
- that these synthetic studies demonstrate a natural-language or large-scale MoE result.

## What the evidence currently favors

The strongest architectural direction is **structured shared subspaces + parallel routing + joint optimization**, not unconstrained per-token support search.

The strongest representation-learning question is no longer whether a Mirror transform can be written. Phase I and early Phase II already contain ample evidence that shared/private parameterizations and task-sensitive modulation are possible. The open question is whether task structure can organize concepts into reusable shared parameter identities and transferable low-description state/view coordinates.

## Next gate

The next experiment should explicitly couple **semantic consistency** and **functional quality** instead of assuming one will emerge from the other.

The leading test is:

- provide multiple related concept families and multiple reusable state/view transformations;
- train shared bases and state/view codes jointly;
- add an objective that requires the *same state/view code* to induce a consistent functional change across different bases;
- compare against additive, CP/low-rank, and independent controls at matched serialized bytes;
- hold out family x state combinations;
- measure task NLL/accuracy, state-swap consistency, cross-family transfer, actual bytes, and semantic clustering;
- separately test the dual parameterization: **shared canonical parameter + Mirror coordinate** versus **shared Mirror direction/basis + concept-specific coordinate**.

A result is only Mirror-specific if it survives strong additive/low-rank controls.

## Evidence-package provenance

Large training artifacts remain outside normal Git history. The following local evidence packages were verified before this summary was written:

| Experiment | Evidence package SHA-256 | Report SHA-256 |
|---|---|---|
| MN003 | `a804ebd3b2c2aef37b864686711a54024105d1149d4eaebb816d9136d4fc2fc4` | `e20210be58e532e5b2c202d45e2d7d0b412893bff44d3b652e2e5ca727031003` |
| MN004 | `270fe2f34c211ad51809a8761a6a4432c0a6b8a91a5725ab95eb18546f5446ea` | `fd75eca0e226f0aae0352c29cbb96361bd83c722ddc07ee8c738e4fc29d3ed12` |
| MN005 | `23c0273a849ec5037f110688c29882ce0e99e9965e2d43a113446576c42488c7` | `f0195e112e402a902dbb95c2af81f8a150b885d85380dc0abdede0c02acd3573` |
| MN006 | `a7c6bd978ad48b2b41624653b46d0fc39d921060542dfa191b91b43ae2144846` | `4b19604578b49f2af57d20a799ef46c506632cdee763af0ed1714daa523b393c` |
| MN007 | `41e6359eeff530b9f57e56b3ed91b887c2cb6dfcb4e3e2c5c0a3f97076d194df` | `2264acfbb11a2d75f53752d86670862eaf5937ed631b6fdc6f2d7d674469d3aa` |

The hashes establish identity of the local evidence bundles; they do not make the omitted binary artifacts available from GitHub.
