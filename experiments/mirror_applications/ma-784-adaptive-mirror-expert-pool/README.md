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

## Results and decision

### T — Execution

The amended frozen screen ran seven conditions for development seeds 78401 and 78402, each for 1,200 updates. Tiny Shakespeare SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. The script read the first 80% of bytes for training and the next 10% for development; the final 10% was not decoded, tokenized, evaluated, or used for selection. The provenance hash was streamed over the complete file. Successful model training wall time summed to 1,033.7 seconds; the logged failed serialization attempt added about 60 seconds and 1,200 updates. PyTorch 2.6.0 CPU, one thread.

| condition | seed 78401 dev NLL | seed 78402 dev NLL | actual payload bytes | expert bank tensor bytes |
|---|---:|---:|---:|---:|
| dense | 2.180885 | 2.166205 | 603,690 | 0 |
| untied MoE | 2.162263 | 2.129338 | 1,423,550 | 1,060,864 |
| UniPool, 4 physical / 4 logical | 2.163068 | 2.189943 | 611,690 | 265,216 |
| UniPool, 2 physical / 2 logical | 2.267118 | 2.281575 | 474,458 | 132,608 |
| 2 physical / 4 logical hard aliases | 2.157682 | 2.261070 | 476,746 | 132,608 |
| 2 physical / 4 logical Mirror | 2.192944 | 2.165198 | 477,356 | 132,608 |
| 2 physical / 4 logical FiLM | 2.185554 | 2.144324 | 477,234 | 132,608 |

All inference byte counts are actual `.pt` serialized payload sizes including model state, vocabulary, config, and metadata. Mirror saved 134,334 bytes (21.96%) against four-expert UniPool and halved the expert-bank tensor bytes, but its payload was 122 bytes larger than FiLM and 610 bytes larger than hard aliasing. The Givens code adds 128 float32 values per model.

The fixed quality/storage gate passed in both seeds: Mirror stayed within +0.10 nat of four-expert UniPool and used at most 95% of its serialized bytes. The Mirror-specific gate failed:

- Seed 78401: Mirror was +0.029876 nat versus four-expert UniPool, +0.007390 versus FiLM (worse), and +0.035261 versus hard aliases (worse).
- Seed 78402: Mirror was -0.024744 nat versus four-expert UniPool, +0.020874 versus FiLM (worse), and -0.095872 versus hard aliases.
- Mirror failed to beat FiLM by the required 0.02 nat in either seed; therefore the locked rule kept audit unopened.

Mean development NLL was 2.179071 for Mirror, 2.164939 for FiLM, 2.209376 for hard aliases, and 2.176506 for four-expert UniPool. Mirror's active MAC proxy was 198,864 per token versus 198,672 for four-expert UniPool. Mean evaluation throughput was 97.6k tokens/s for Mirror and 98.0k for FiLM on this CPU implementation. These are small-run measurements, not optimized serving results.

### D — FAIL

The preregistered Mirror-specific gate failed even though pool storage and the quality/storage screen passed. No audit or fresh data were opened. This is a fixed-budget development result, not a capacity claim.

### C — Strongest counter-hypothesis

FiLM provides the needed route specialization more simply and is byte-near; the two-seed measurements support this. Hard route aliases also matched or beat Mirror inconsistently, suggesting much of the benefit came from the routing parameterization rather than the Givens coordinates.

### U — Unconfirmed

Near-convergence quality, other data/worlds, larger models, optimized inference, and whether a different Mirror coordinate or private residual changes the frontier remain untested.

### Fact / interpretation / hypothesis

- **Fact:** The 2-physical/4-logical Mirror payload is 21.96% smaller than the 4-physical UniPool payload and uses half the expert-bank tensor bytes. It passes quality/storage gates in both development seeds but loses to byte-near FiLM in both.
- **Interpretation:** In this fixed-budget screen, `m` does not add Mirror-specific value beyond the simple gate; the Givens view also adds a small compute cost.
- **Hypothesis:** Expert-pool shrinkage may still be useful when paired with another native conditioner or when private residual capacity is allocated selectively. This experiment does not establish that.
