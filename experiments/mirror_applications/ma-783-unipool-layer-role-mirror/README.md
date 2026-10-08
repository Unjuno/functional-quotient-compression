# MA-783 — UniPool global expert pool + Mirror layer role

Status: **SCREENING — protocol frozen before corpus access**
Branch: `research/ma-783-unipool-layer-role-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 13 selected **MA-783** uniformly from 555 eligible P0/UNTESTED candidates. Ordered pool, SHA-256, exclusions, seed and index are under `source/`.

## H — Falsifiable hypothesis

One global pool of four physical FFN experts plus eight per-layer Givens coordinates can recover enough layer-specific function to approach a four-layer layer-owned MoE on held-out character-language NLL at no more than 70% of its full serialized inference bytes, and outperform byte-near FiLM and native depth-embedding controls.

## Prior art

PA205 UniPool already shares one global expert pool across layers with independent routers, balancing and scale-stable routing. PA204 MoRE reuses experts across layer groups and uses learned depth embeddings. This experiment tests the marginal effect of a compact orthogonal layer View, with both a same-size simple gate and a depth-embedding control. It does not claim expert sharing or depth conditioning as new.

## T — Frozen experiment

The protocol specifies a 4-layer, width-64 causal character Transformer, four experts, two development seeds, 1,200 updates, six conditions, quality/bytes and Mirror-specific gates. The corpus will be acquired only after the protocol commit. Audit spans remain locked unless all development gates pass.

## D — Pending

No corpus values or model scores have been accessed.

## C — Strongest counter-hypothesis

UniPool itself may already capture nearly all useful shared expert behavior, while learned depth embeddings or simple FiLM gates provide equal or better specialization with similar bytes.

## U — Unknown

Whether the layer View preserves useful NLL, expert/load diversity, true end-to-end bytes and CPU compute remains unmeasured.
