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

See `PROTOCOL.json`; fresh seeds are sealed. Required controls: independent cache, MLKV-style hard sharing, shared-basis direct coefficients, Mirror coordinates, and private residual upper control.

## Results

Pending development.
