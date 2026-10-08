# MA-464 — AdaMix over logical Mirror adaptations

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Selection: Draw19, uniform over eligible P0/UNTESTED candidates; replay in `source/random_draw.json`.

## H — Hypothesis

One rank-4 QKV LoRA basis with two learned latent-space Givens View codes, sampled stochastically during adaptation, can preserve the per-view and merged quality of AdaMix's two independent rank-4 modules across two text domains while using fewer actual serialized inference bytes; the structured Views should beat an equal-byte shared-basis FiLM control.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-view rank-space rotation `m` between one shared LoRA A/B basis on each nanoGPT attention QKV projection, so two logical adaptation views can be trained with the stochastic route schedule of AdaMix without storing two independent A/B pairs.

- Physical object: rank-4 low-rank updates to the QKV projection in every layer of one frozen nanoGPT base.
- Native method: AdaMix-style stochastic selection among two independent LoRA branches; mean-update merge at inference.
- Mirror coordinate: six Givens angles per layer and logical view, applied in the rank-4 latent space.
- Logical functions: two adapter views plus their merged single-module deployment.
- Closest prior art: PA83 AdaMix; the native stochastic training and merged inference controls remain primary.
- Simple controls: base only, ordinary single LoRA, shared rank-4 basis with per-view diagonal FiLM scales.

## T — Frozen experiment

The common base is a four-layer, width-64 nanoGPT character model pretrained on the Shakespeare training span and then frozen. Adaptation uses a 50/50 mixture of Shakespeare and Pride and Prejudice character streams. Conditions are base-only, one rank-4 LoRA, two independent rank-4 AdaMix branches, one shared rank-4 basis with two Mirror rotations, and the same shared basis with two FiLM scale codes. At each adaptation update, AdaMix/Mirror/FiLM sample one logical branch/view uniformly for the minibatch. Development NLL is measured separately on both domains and after exact mean-update merge.

The exact source URLs, domain splits, seeds, updates, loss, byte accounting, compute measures, gates, and sealed audit rule are in `PROTOCOL.json`. This is a two-domain low-resource mechanism screen; it cannot establish general PEFT quality or model capacity.

## D — Decision gates

Mirror must, in both seeds, stay within +0.10 mean dev nat of AdaMix for per-view and merged inference, use at most 99% of AdaMix's complete two-view inference-bank bytes and at most 55% of its adapter-only bytes, and beat the matched-byte FiLM control by at least 0.02 mean dev nat for both per-view and merged quality. Audit opens only if every development gate passes in both seeds.

## C — Strongest counter-hypothesis

The independent AdaMix branches may need distinct A/B subspaces that cannot be represented by rotating one shared rank-4 basis. If Mirror matches the simpler FiLM control, any gain is shared-basis conditioning rather than Mirror-specific value.

## U — Open questions

The natural-text domain gap may be too small or too large for the frozen base; stochastic batch routing may not match the paper's exact schedule; the two View codes may collapse; merged quality may not preserve per-view behavior; complete model bytes may be dominated by the frozen base. These are measured or reported as boundaries, not tuned away.
