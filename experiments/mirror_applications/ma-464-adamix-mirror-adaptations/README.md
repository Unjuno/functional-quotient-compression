# MA-464 — AdaMix over logical Mirror adaptations

Status: FAIL (frozen development gates)
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

## Outcome

**Fact.** Both frozen seeds completed. For each seed Mirror's complete two-view bank is 897,778 bytes versus AdaMix's 912,610 bytes (1.63% smaller), while the adapter-only payload is 21,202 bytes versus 36,232 bytes (41.48% smaller, above the preregistered 55% ceiling of AdaMix, 19,927.6 bytes). Mirror stays within +0.10 nat of AdaMix for view and merged NLL in both seeds. Its full bank is within 5% of FiLM's 897,386 bytes. Mirror's mean view / merged dev NLL are 2.256782 / 2.256640 (seed 46401) and 2.208794 / 2.208705 (seed 46402); FiLM is 2.253763 / 2.253698 and 2.207594 / 2.207571. Thus Mirror does not beat FiLM by the required 0.02 nat in either seed. Development gates failed and audit remained unopened.

**Interpretation.** The learned Givens coordinates produced distinct logits (mean absolute view distances 0.02663 and 0.02103), but these views did not add measurable quality over the byte-matched diagonal FiLM control. They did reduce full-bank storage slightly relative to AdaMix, but failed the stricter adapter-only threshold.

**Hypothesis.** The shared rank-4 basis constrains view diversity; the near-identical FiLM outcome suggests the observed adaptation is explained by generic per-view conditioning. More seeds or domains could change the estimate, but are outside this frozen screen.

**H** The falsifiable claim was that a shared rank-4 basis plus two learned Givens views would match AdaMix within +0.10 nat, satisfy both byte limits, and beat byte-matched FiLM by 0.02 nat on both view and merged inference. **T** Two fresh initializations (46401, 46402), one Shakespeare-pretrained frozen 4-layer width-64 nanoGPT per seed, 600 adaptation updates per condition over Shakespeare/Pride and Prejudice, with single LoRA, two-branch AdaMix, Mirror and FiLM controls; all actual payloads were serialized and measured. **D: FAIL.** **C** FiLM or generic shared-basis conditioning explains the results; the 0.02 nat Mirror-specific margin is absent. **U** No audit/generalization claim, near-convergence claim, or capacity claim is established.

## D — Decision gates

Mirror must, in both seeds, stay within +0.10 mean dev nat of AdaMix for per-view and merged inference, use at most 99% of AdaMix's complete two-view inference-bank bytes and at most 55% of its adapter-only bytes, and beat the matched-byte FiLM control by at least 0.02 mean dev nat for both per-view and merged quality. Audit opens only if every development gate passes in both seeds.

## C — Strongest counter-hypothesis

The independent AdaMix branches may need distinct A/B subspaces that cannot be represented by rotating one shared rank-4 basis. If Mirror matches the simpler FiLM control, any gain is shared-basis conditioning rather than Mirror-specific value.

## U — Open questions

The natural-text domain gap may be too small or too large for the frozen base; stochastic batch routing may not match the paper's exact schedule; the two View codes may collapse; merged quality may not preserve per-view behavior; complete model bytes may be dominated by the frozen base. These are measured or reported as boundaries, not tuned away.
