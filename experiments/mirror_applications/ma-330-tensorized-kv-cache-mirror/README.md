# MA-330 — Tensorized KV cache/view reconstruction

Status: protocol frozen before development. Dedicated branch: `research/ma-330-tensorized-kv-cache-mirror-20261008`.

## H — hypothesis

A shared temporal K/V basis plus small layer/head Mirror coordinates can reconstruct aligned logical cache views with low held-out attention-output error and reduce actual payload bytes; decode-time reconstruction may erase the storage gain.

## Mirror insertion

> **Mirror insertion:** add compact layer/head coordinates at the shared temporal-basis-to-cache reconstruction boundary, so logical layer/head K/V state can be represented without storing every full cache independently.

- Native method: independent KV cache or MLKV hard sharing across layers.
- Insertion point: per-layer/head reconstruction from shared K/V temporal factors.
- `m`: persistent address stored with cache state.
- Nearest ordinary control: flat direct coefficients over the same paid basis; also MLKV hard sharing.

## Prior art

PA08 (MLKV) shares KV heads across layers. PA35 motivates shared tensor factors but does not establish dynamic-cache reconstruction. This experiment measures cache payload, output quality, and reconstruction cost separately.

## Frozen controls and gates

See `PROTOCOL.json`; fresh seeds are sealed. Required controls: independent cache, MLKV-style hard sharing, direct cos/sin coefficients with private unrelated-layer state, Mirror phase codes with private state, and a no-private Mirror condition to expose the private-state boundary.

The development implementation was frozen before fresh access. Its timing now reports cache reconstruction and causal attention separately, with total query throughput including reconstruction. This timing amendment does not change thresholds or seeds.

## Results

Pending development.

Development also reports per-layer attention-output error. The fourth layer was planted unrelated; the no-private condition should expose whether one shared cache address can serve it. Frozen development suggests it cannot, while one private cache restores that layer.
