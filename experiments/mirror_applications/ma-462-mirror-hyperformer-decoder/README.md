# MA-462 — Fixed-embedding Mirror decoder vs HyperFormer-style decoder

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-462-mirror-hyperformer-decoder-20261008`  
Base commit: `d12d62a1902c702e57b09cfaba49d0862d485226`  
Prior art: PA82 and PA118.

## H — hypothesis

With identical fixed task-factor embeddings and a shared frozen backbone/adapter basis, a structured Mirror decoder can generate useful adapters for held-out task-factor combinations with lower full inference bytes than a generic HyperFormer-style decoder, and can beat simple linear/additive decoders when task variation is factor-aligned.

> **Mirror insertion:** this experiment adds `m(a,b)` to the shared adapter-atom coefficient interface so task-specific logical adapters can be generated from two factor coordinates without storing a separate adapter decoder or full adapter for every task.

## Prior-art delta and controls

PA82/PA118 already establish shared task-conditioned hypernetworks that generate adapters. The tested delta is whether a structured Mirror code is smaller or extrapolates better than that generic decoder while using the same fixed conditioning embeddings. The generic hypernetwork is a small coefficient-emitting HyperFormer-style control, not a claim of reproducing the full paper architecture.

Controls are frozen base/no adapter, linear decoder, generic shared hypernetwork, additive factor decoder, flat per-task coefficients, Mirror plus private residual, and independent full adapters.

## Protocol

- A task-agnostic MLP is pretrained on a common regression teacher for 500 updates, then frozen. Adapter atoms attach to its output projection.
- Sixteen tasks form a 4×4 factor grid. Four combinations are held out while all factor values remain represented in training.
- Fixed `E_A` and `E_B` embeddings are generated per world and held identical across every decoder. They are never trained.
- Target task adapter code: `c(α)=α c_view(a,b)+(1−α)c_private(task)`, with α in `(0, 0.25, 0.5, 0.75, 1)`. `c_view` is product-factorized; `c_private` is independent per task.
- Development seeds 4621/4622; fresh seeds 46201/46202/46203. Decoder training is fixed at 300 updates. Mirror/additive latent ranks `(2, 4, 8)` are selected using development only.
- The alpha=1 development gate requires both worlds to be within 10% of HyperFormer, at least 20% better than the best linear/additive control on held-out tasks, and at most 80% of HyperFormer's complete serialized payload. If no rank passes, fresh remains unopened.
- Actual `torch.save` inference bytes include frozen base, adapter atoms, fixed embeddings, all decoder state and metadata. Decoder MACs, adapter application MACs, training proxy, generator wall time and inference wall time are separate.

## Random selection provenance

Draw 4 selected MA-462 from 531 registered P0/UNTESTED candidates with no remote `research/ma-*` branch. Seed, index, pool hash and exact eligible list are in `source/random_draw.json` and `source/selection_pool.csv`.

## Decision

FACT: pending.

INTERPRETATION: pending.

HYPOTHESIS: pending.

BOUNDARY: synthetic decoder mechanism only; no natural-language, pretrained LLM or full HyperFormer++ claim.
