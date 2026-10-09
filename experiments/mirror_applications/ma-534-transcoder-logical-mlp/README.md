# MA-534 — Role-specific logical MLP views over a shared transcoder bank

## H — Hypothesis

Four fit-derived context roles can use role-specific 16-angle Givens views over one shared 32-atom transcoder bank to reduce held-out raw layer-8 MLP output error by at least 10% relative to both plain role-specific top-8 sparse gates and diagonal per-atom gain controls.

## T — Frozen setup

Pinned SmolLM2-135M and EleutherAI layer-8 skip-transcoder, Wikitext-2 raw train split, 32 non-overlapping 128-token blocks, 16 fit and 16 evaluation. Four nearest-centroid roles are learned on fit inputs only. A shared 32-feature bank is chosen on fit activations. Every method uses the same bank and top-8 decode. Only Mirror angles and the diagonal control's gains receive 200 fit updates. See `PROTOCOL.json` for all fixed details.

## D — Pending

The development run has not started. Fresh validation remains sealed until every frozen development gate passes.
