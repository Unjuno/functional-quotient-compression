# Worker queue

The queue is derived from `IDEA_REGISTRY.csv`. The registry is authoritative.

## Selection rule

Pick the first candidate satisfying all of:
1. status is UNTESTED;
2. highest available priority;
3. no other active experiment directory already claims the ID;
4. its closest prior-art controls can be implemented in the current harness.

Do not skip to a visually interesting P1/P2 idea while an executable P0 remains, unless the skipped candidate has a recorded blocker.

## Literature-derived cross-over queue

These were added after the 2026-10-07 prior-art sweep and should be considered before duplicating a simpler P0 experiment:

1. MA-241 — expert tying across depth + layer-specific Mirror expert views
2. MA-253 — cache-safe final-layer Mirror-MoE
3. MA-244 — K=V projection sharing + Mirror role recovery
4. MA-245 — MLKV shared cache + per-layer Mirror KV views
5. MA-247 — recursive shared block + Mirror depth modulation
6. MA-248 — packet random variable as a Mirror code
7. MA-249 — one physical future head + Mirror future-offset views
8. MA-250 — MAP/Hadamard binding as Mirror expert address
9. MA-251 — factorized expert x depth Mirror coordinate

These have strong adjacent prior art, so the experiment must implement the cited non-Mirror method as a control.

## Second research-expansion queue

Added from the second literature sweep. **Do not interrupt an already-started experiment to switch queues.**

Direct/high-information P0 order:
1. MA-255 — Mirror context superposition vs Parameter Superposition
2. MA-260 — BatchEnsemble rank-one Mirror ensemble
3. MA-261 — BatchEnsemble-style logical experts
4. MA-265 — VeRA Mirror scaling code bank
5. MA-268 — IA3 Mirror activation views
6. MA-271 — OFT Mirror task views
7. MA-272 — input-centric OFTv2 Mirror views
8. MA-273 — BOFT Mirror adapter bank
9. MA-274 — BOFT logical expert views
10. MA-276 — BOFT depth views
11. MA-278 — Compacter Mirror hypercomplex adapters
12. MA-282 — Monarch Mirror FFN transform
13. MA-286 — Cheap-LoRA Mirror column-subspace views
14. MA-288 — Mirror fast-weight programmer context code
15. MA-292 — task-vector Mirror basis
16. MA-296 — orthogonalized task-vector Mirror superposition
17. MA-297 — SETA shared sparse subspace + Mirror views
18. MA-299 — Split-on-Share Mirror code allocation

Then continue the remaining P0 entries by family and registry order.

## Current P0 sequence

### Family A — FFN / MoE / adapter
MA-003 -> MA-005 -> MA-009 -> MA-019 -> MA-024

### Family B — Attention / KV
MA-041 -> MA-048 -> MA-061 -> MA-063

### Family C — Depth / position
MA-076 -> MA-079 -> MA-086 -> MA-116

### Family D — Temporal
MA-121 -> MA-129

### Family E — Compression / holographic
MA-156 -> MA-160 -> MA-171 -> MA-173 -> MA-181

### Family F — Continual / optimization / distillation
MA-186 -> MA-189 -> MA-199 -> MA-208

## Family handoff rule

Within one family, reuse the same parent checkpoint, serializer, data split, benchmark code and result schema whenever scientifically valid.

A worker may batch implementation work across a family, but scientific status is still assigned per MA ID.

## Stop rule

If two consecutive candidates in a family fail for the same demonstrated structural reason, stop that family and write a family diagnostic before continuing.
