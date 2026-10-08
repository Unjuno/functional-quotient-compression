# MA-303 — Factorized layer-mask Mirror codes

Status: protocol frozen before development. Dedicated branch: `research/ma-303-factorized-layer-mask-mirror-20261008`.

## H — hypothesis

A factorized Mirror address for task x layer mask combinations will recover held-out factor pairs on a fixed random two-layer ReLU network and provide a material actual-byte improvement over Piggyback masks and a direct shared-factor coefficient control. Unrelated masks should require private state.

## Mirror insertion

Native method is Piggyback-style binary task masks over a fixed random backbone. Mirror inserts a rank-2 factor interaction into the edge mask, generated from task and layer coordinates on a shared basis. The cheapest ordinary control is a direct real-valued rank-2 task x layer coefficient bank; independent masks are the upper control.

## Frozen protocol

See `PROTOCOL.json`. Six task factors and six layer factors are observed during development/training. Four combinations from held-out factor IDs are tested in fresh worlds. Each pair has 256 support, 128 validation and 256 test examples. Seeds: development 30301/30302; fresh 30311/30312/30313. No optimizer updates. Actual fixed-timestamp ZIP/NPY bytes are authoritative.
