# MA-784 — Adaptive global expert pool with logical Mirror views

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Selection: Draw18, uniformly sampled P0/UNTESTED row; replay is in `source/random_draw.json`.

## H — Hypothesis

With four logical routes and top-1 activation, a global pool of two physical experts plus layer-and-route Givens views can match a four-expert UniPool at lower actual payload bytes and beat both hard aliasing and a byte-matched FiLM gate on held-out character-LM NLL.

## Mirror insertion

> **Mirror insertion:** this experiment adds per-layer/per-logical-route Givens coordinate `m` to the input of one selected expert in a shared global expert pool, so four logical route behaviors can be represented by two physical expert weight sets without storing four physical experts.

- Physical object: global expert FFN weights shared across four Transformer layers.
- Exact insertion: after top-1 routing, apply eight learned Givens rotations to the first sixteen channels of the selected expert input; each logical route has a code for each layer.
- Logical multiplicity: four logical expert routes over a two-expert physical pool.
- Native methods: PA205 UniPool global pool and ordinary MoE.
- Cheapest controls: hard route aliases with no view; same-shape FiLM scales (8 groups × 4 routes × 4 layers).
- Prior-art delta: test whether `m` lets the physical pool shrink from four to two at fixed logical route count, beyond existing UniPool pooling and ordinary FiLM.

## T — Frozen execution

Small 4-layer, width-64 character Transformer; Tiny Shakespeare public-domain corpus; top-1 NormRouter and global pool-level load balance for all global-pool conditions. Conditions are dense, untied four-expert-per-layer MoE, native four-expert UniPool, two-physical/two-logical UniPool, two-physical/four-logical hard aliases, two-physical/four-logical Mirror Givens, and two-physical/four-logical FiLM. Two development model seeds, fixed 1,200 updates, shared contiguous train/dev split. Fresh audit span remains unopened unless the development gates pass in both seeds.

The exact architecture, seeds, split rule, optimizer, checkpoints, gates, storage objects, compute accounting, and audit rule are frozen in `PROTOCOL.json`. This is a small fixed-budget screen, not a capacity claim.

## D — Decision gates

PASS requires in both development seeds: Mirror NLL ≤ native four-expert UniPool +0.10 nat; Mirror full serialized payload ≤95% of four-expert UniPool; Mirror NLL improves by at least 0.02 nat over both byte-near FiLM and hard-alias controls; Mirror expert-bank bytes are lower than the four-expert pool; no failed-run exclusion. Audit opens only if every development gate passes. Otherwise record FAIL and leave audit unopened.

## C — Strongest counter-hypothesis

UniPool's shared routers and balancing provide most of the reuse benefit; when the physical pool shrinks, four logical labels do not restore independent expert functions. Any apparent gain may be reproduced by ordinary FiLM or hard aliases, while the total payload is dominated by the Transformer trunk.

## U — Boundaries

The run can establish only a fixed-budget small character-LM mechanism signal. It cannot establish asymptotic capacity, production-scale expert reuse, arbitrary expert multiplicity, or optimized serving. Count actual serialized bytes for all tensors, routing state, view/FiLM codes, vocabulary, configuration, and metadata. Report active MAC proxy and wall time separately.

## Execution note

The first development attempt completed 1,200 updates for the dense condition, then failed while serializing metadata because the no-router baseline had no NormRouter calibration entry. No metric row or checkpoint was retained. The protocol records this attempt and a serializer-only correction (`normrouter_c=null`, `routing=none`) before an identical rerun. All model/data/split/seed/update/metric/gate settings remain frozen.
