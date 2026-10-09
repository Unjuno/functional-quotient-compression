# MA-457 — Path reuse before physical module birth

Status: **FAIL under quality, byte, and compute gates; scoped birth reduction observed**  
Evidence lane: CONTINUAL PATH REUSE / RETENTION / MODULE BIRTHS / BYTES / COMPUTE  
Protocol frozen: `b89b64fd`; fresh worlds 45710–45712.

## H — Hypothesis

A support-selected Givens View of a frozen PathNet module can reuse it across recurring transformed tasks and avoid physical module births while preserving query quality and retention better than path-only reuse.

## T — Test

Synthetic sequential 2D linear tasks: task 0 defines a canonical matrix; tasks 1–23 are output rotations of that matrix; tasks 24–31 are off-orbit perturbations. Each task had 32 support points, 32 route-validation points, and 256 queries. Compared path-only module reuse, Mirror-angle reuse, and an always-private module. Development selected validation threshold 0.1 from `{0.01,0.03,0.1}` under the preregistered mean-quality constraint. Fresh evaluation used three worlds × three seeds × 32 tasks. Actual serialized modules, route IDs, and angles were charged at N=1/8/16/32.

## D — FAIL; scoped module-birth reduction

**Fact:** At N=32, mean module births were 7.11 Mirror vs 24.22 path-only vs 32 private. Mirror NRMSE was 0.00769 vs 0.01522 path-only and 1.42e-7 private. Payload was 71.82B/task Mirror, 80.27B path-only, and 69.28B private. Thus Mirror reduced births but saved only 10.5% payload vs path-only, far short of the registered 25%, and its error was far above private. Support route search used 4.99M MAC/sequence for Mirror vs 0.10M for path-only; mean search wall was 0.610s vs 0.0127s. Query wall was also higher. Prior 24-task Mirror retention NRMSE after the sequence was 0.00424; frozen modules had no destructive overwrites.

**Interpretation:** Mirror Views help reuse a module across the deliberately rotated task cohort and cut births, but route search costs roughly 50× the path-only MAC proxy, and payload/quality do not beat private state. This is a scoped reuse mechanism, not a validated continual-learning compression win.

## C — Strongest counter-hypothesis

The task family is deliberately aligned to output rotations, while the eight off-orbit tasks still require new modules. The birth reduction may not transfer to natural task streams, and ordinary cached path selection could recover part of the routing cost.

## U — Unknown

Natural task sequences, higher-dimensional learned modules, non-orbit detection, cache-hit behavior, and retention under online module updates remain untested.
