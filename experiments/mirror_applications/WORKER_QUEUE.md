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

## Third research-expansion queue — representation and symmetry

Added after the subnetwork/tensor/symmetry sweep. Do not interrupt active work.

High-information P0:
1. MA-301 — continuous Mirror supermask
2. MA-307 — Mirror code before PackNet physical allocation
3. MA-311 — Mirror task code in intrinsic subspace
4. MA-312 — shared intrinsic basis + many Mirror task coordinates
5. MA-314 — adaptive intrinsic-dimension allocation
6. MA-319 — Tucker matrix-bank Mirror layer coefficients
7. MA-320 — Tucker logical experts
8. MA-322 — TT-core Mirror adapter bank
9. MA-325 — tensorized embedding domain views
10. MA-327 — factorized layer x expert Tucker address
11. MA-330 — tensorized KV reconstruction
12. MA-331 — Re-Basin-aligned Mirror task deltas
13. MA-332/333 — permutation and sign/scale symmetry audits
14. MA-338 — symmetry-normalized Mirror code learning
15. MA-341/342 — personalized/federated Mirror codes
16. MA-344 — PreLort nested-rank Mirror segments
17. MA-349 — Rank-1 Bayesian Mirror posterior
18. MA-351 — MIMO + Mirror diversity
19. MA-355/356 — product-key Mirror addresses
20. MA-357 — Hopfield reservoir for Mirror addresses
21. MA-359 — ACDC/AFDF Mirror transform
22. MA-360 — reversible Mirror block

## Fourth research-expansion queue — execution structure and module banks

Added after the dynamic-compute/supernet/prompt/embedding sweep. Do not interrupt active work.

High-information P0:
1. MA-361 — Mixture-of-Depths + Mirror block-role view
2. MA-364 — early-exit Mirror readout views
3. MA-366 — factorized depth x expert routing
4. MA-367/368 — slimmable width/depth Mirror codes
5. MA-369 — Once-for-All subnetwork Mirror correction
6. MA-371/372 — MatFormer granularity and Mix'n'Match Views
7. MA-374 — ALBERT shared layers + depth Mirror
8. MA-375 — one-shot supernet + Mirror correction
9. MA-379 — Mirror-compressed AdapterFusion bank
10. MA-381 — LoRAHub over Mirror-compressed basis
11. MA-383 — L2P prompt pool + Mirror generator
12. MA-385 — DualPrompt expert prompts as Views
13. MA-389 — Hash Embedding Mirror importance codes
14. MA-391/392 — compositional embedding Mirror addresses
15. MA-393 — adaptive-capacity embedding + View
16. MA-395 — ALBERT factorized embedding + domain View
17. MA-397 — product-address Mirror vocabulary
18. MA-399 — MatFormer speculative drafter via View

## Fifth research-expansion queue — conditional functions and dynamics

Added after the conditional-modulation/dynamical-system/meta-learning sweep. Do not interrupt active work.

High-information P0:
1. MA-401 — FiLM versus Mirror feature conditioning
2. MA-403 — token-wise generated Mirror modulation
3. MA-405 — StyleGAN2-like FFN weight modulation
4. MA-407 — demodulated Mirror-MoE
5. MA-408 — CondConv-style synthesized FFN
6. MA-411 — canonical transform + sparse Mirror refinement
7. MA-413 — factorized concept Mirror coordinates
8. MA-416/417 — shared decoder + Mirror function codes
9. MA-418/419 — compositional/modulated neural-function codes
10. MA-424/425 — continuous-depth Mirror dynamics
11. MA-427 — DEQ conditioned fixed-point map
12. MA-429/431 — Universal Transformer depth Views/composition
13. MA-434 — Mamba selective-state Mirror roles
14. MA-436/437 — logical SSM experts and S4 structured Views
15. MA-442 — MAML with Mirror-only inner-loop adaptation
16. MA-444 — LEO latent decoder versus structured Mirror
17. MA-446 — learned optimizer for Mirror coordinates

## Sixth research-expansion queue — modular programs, editing and coded addresses

Added after the modular-routing/model-editing/codebook sweep. Do not interrupt active work.

High-information P0:
1. MA-451/452 — PathNet path + Mirror role/factorization
2. MA-453 — Routing Network with logical Mirror blocks
3. MA-455 — sequential Mirror program over one block
4. MA-457 — path reuse before module birth
5. MA-461/462/463 — HyperFormer versus generated Mirror adapters
6. MA-464 — AdaMix over logical Mirror adaptations
7. MA-466 — UniPELT components as Mirror axes
8. MA-468 — Polytropon shared/private skill bank + Views
9. MA-469/470 — MEND-generated Mirror edit codes
10. MA-471 — ROME rank-one edit as Mirror coordinate
11. MA-473 — MEMIT edit basis + Mirror memory codes
12. MA-475/476 — SERAC/GRACE edit-memory compression
13. MA-478 — compact View first, explicit edit fallback
14. MA-481/482 — VQ and residual-VQ Mirror addresses
15. MA-484 — VQ logical expert codebook
16. MA-486/487 — sparse dictionary Mirror functions + LISTA routing
17. MA-488 — shared/private dictionary + Mirror coefficients
18. MA-492 — quantized packet-plan latent
19. MA-494 — error-correcting Mirror expert IDs
20. MA-498 — learned code-distance regularization

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
