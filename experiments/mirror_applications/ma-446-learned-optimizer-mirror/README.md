# MA-446 — learned optimizer restricted to Mirror coordinates

Status: **SCREENING — protocol frozen before development**
Branch: `research/ma-446-learned-optimizer-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Random selection: Draw 10, 558 eligible P0/UNTESTED rows; pool, hash, seed and index are in `source/random_draw.json`.

## H — falsifiable hypothesis

On a related few-shot task family represented by one shared matrix basis and a three-value Mirror code, a shared learned recurrent optimizer that updates only `m` will beat tuned SGD/Adam on that same code after four support updates, stay within 5% of a full-weight learned optimizer, and use at least 30% fewer serialized restartable bytes for eight tasks.

## Prior-art delta

PA77 (Andrychowicz et al., *Learning to learn by gradient descent by gradient descent*) learns a recurrent optimizer from gradient history. PA151 (Meta-SGD) is the simpler learned initialization and per-coordinate update-scale control. This screen tests applying one recurrent learned update rule only to a compact Mirror code, versus SGD/Adam/Meta-SGD on the same code and the same recurrent optimizer architecture over full weights.

## Frozen protocol

See `PROTOCOL.json`. The task family is synthetic linear regression with 4 input, 3 output dimensions, a shared weight matrix, a fixed rank-3 basis, eight logical tasks, 16 support examples per update, four updates, and 128 query examples. Two development worlds are seeded 44601 and 44602. Only SGD/Adam learning rates may be tuned on development. Fresh families remain locked until every gate passes.

Controls include hard sharing, SGD/Adam on `m`, learned recurrent optimization on `m`, full-weight Adam, and the same recurrent optimizer on all 12 weights. The full optimizer is the capacity upper control. All optimizer weights, per-task code/state, basis and metadata are charged in actual serialized eight-task payload bytes.

## D — pending

No development scores have been inspected. Implementation and the frozen development run follow this protocol.

## C — strongest counter-hypothesis

Adam's gradient normalization may already handle the small Mirror code efficiently; a recurrent learned optimizer may add more shared bytes and compute without improving few-shot query quality.

## U — unknown

Held-out adaptation quality, optimizer transfer, actual restartable library bytes, and the update-compute/runtime tradeoff remain unmeasured.
