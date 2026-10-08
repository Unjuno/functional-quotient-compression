# MA-760 — diffusion control View + private residual

Status: SCREENING; protocol frozen before development scores.  
Evidence lane: mechanism / storage / compute / runtime.  
Branch: `research/ma-760-diffusion-control-view-20261008`.  
Base: `407ca7e2047326d1e4b753e55e05c4730f26f32b`.  
Draw 6 selected this P0 candidate uniformly from 529 eligible IDs; provenance and full pool snapshot are in `source/random_draw.json` and `source/selection_pool.csv`.

## Hypothesis

On structured spatial-control residual tasks, a shared adapter basis with condition-specific Mirror coordinates and a sparse private residual can retain held-out residual fidelity with fewer total bytes than independent condition adapters. The required private state should grow as task heterogeneity grows.

## Prior-art delta

PA190 (ControlNet) duplicates trainable control branches; PA191 (T2I-Adapter) is the cheap condition-specific adapter control. MA-760 asks where shared control behavior stops being adequate and private exceptions become necessary. This screen uses the pinned `hf-internal-testing/tiny-stable-diffusion-pipe` UNet as an integration substrate. It is an internal test fixture, not a quality pretrained generator; synthetic spatial control operators replace real edge/depth/pose training data. It cannot support an image-generation quality claim or a ControlNet/T2I-Adapter reproduction.

## Frozen comparisons

No control, one hard-shared adapter, independent per-condition 3×3 convolutional adapters, shared additive low-rank basis, multiplicative Mirror coordinates over the same basis/code budget, and Mirror plus rank-1 private residual. Two development seeds, three heterogeneity levels, six basis-training conditions, two held-out condition identities, ranks 1/2/4/8. The exact training, gates and fresh-opening rule are in `PROTOCOL.json`.

Whole-library payload bytes include the tiny UNet state, shared basis, codes, private state and metadata. This keeps any adapter-only storage reduction separate from the actual serialized inference artifact. Fresh worlds remain locked unless a development point passes all predeclared gates.

## Protocol amendment before development

The original draft required a 5% reduction in whole-library bytes. Before any development score, this was amended to require an actual strict reduction plus exact byte and percentage reporting, while retaining the <=0.50x per-condition state condition. The tiny shared UNet dominates its toy adapter payload, making a 5% whole-bundle reduction unreachable by construction. No quality or Mirror-specific threshold was relaxed.
