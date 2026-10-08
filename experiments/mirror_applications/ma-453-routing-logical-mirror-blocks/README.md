# MA-453 — Dynamic Routing Network with logical Mirror blocks

Status: **FAIL** (development screen; fresh seeds sealed)  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-453-routing-logical-mirror-blocks-20261008`  
Base commit: `c6bc140`  
Prior art: PA81 (Routing Networks)

## H — Hypothesis

Given two shared physical linear blocks selected by a context router, a two-angle role address would recover four useful module × role functions, lowering held-out RMSE by at least 30% versus routed blocks alone while keeping actual payload within 1.25× and inference compute within 1.25× the rank-2 residual control. A direct native Givens control tests whether the result is Mirror-specific.

## T — Protocol and execution

The frozen task is four-dimensional synthetic linear regression: two modules, two role cues per module, Gaussian noise σ=0.02, 180 optimizer updates, eight examples per update (1,440 per condition), and 64 held-out examples per function (256 total). Development seeds were 45301 and 45302. The fixed router selects a module from a one-hot context cue; it is deterministic and serialized, **not learned**. Conditions were shared block, routed two-block baseline, routed + Mirror Givens, native Givens, rank-2 residual, role-vector residual, and four independent blocks. Inference payloads are actual uncompressed `.npz` files including model and routing metadata. Fresh seeds 45311–45313 stayed sealed because frozen gates failed.

## Results (facts)

| Seed | Method | Held-out RMSE | Payload bytes | Ops/example | Train wall (s) |
|---:|---|---:|---:|---:|---:|
| 45301 | routed blocks | 0.630205 | 1,252 | 8 | 0.3950 |
| 45301 | routed + Mirror | 0.099472 | 1,534 | 12 | 1.1379 |
| 45301 | native Givens | 0.099472 | 1,534 | 12 | 1.3007 |
| 45301 | role-vector residual | 0.070452 | 1,544 | 12 | 0.5459 |
| 45301 | independent blocks | 0.045487 | 1,286 | 8 | 0.5832 |
| 45302 | routed blocks | 0.267377 | 1,252 | 8 | 0.3948 |
| 45302 | routed + Mirror | 0.170017 | 1,534 | 12 | 1.1689 |
| 45302 | native Givens | 0.170017 | 1,534 | 12 | 1.3898 |
| 45302 | role-vector residual | 0.090093 | 1,544 | 12 | 0.5201 |
| 45302 | independent blocks | 0.027477 | 1,286 | 8 | 0.4816 |

Payload SHA-256 and full metric rows are in `RESULTS_CORE.csv` and each seed's `metrics.json`. `verification_report.json` confirms exact serialization round-trip and replay; Mirror and native Givens have identical payload hashes/bytes and RMSE on both seeds. The measured query timings are tiny single-process batches and are not treated as stable runtime evidence.

## D — Decision

**FAIL.** Mirror passes the frozen per-seed quality-vs-routing and byte ratio gates (1.226× routed bytes) and uses 12 versus 16 proxy ops/example for rank-2. However, it exactly aliases the native Givens conditioner, so attribution fails. The simpler full role-vector residual improves RMSE on both seeds for only 10 additional bytes. Four independent logical blocks are both more accurate and smaller in payload than Mirror. Thus this screen does not establish Mirror-specific gains or a Pareto improvement over strong controls.

## C — Strongest counter-hypothesis

The aligned role-conditioned function is easy to learn because the generator itself applies a Givens rotation. Any direct Givens parameterization recovers the same solution, while a small role table is more flexible and the independent bank provides still better quality per byte. Apparent benefit against routing-only is an extra role degree of freedom, not a Mirror-specific effect.

## U — Unconfirmed

This synthetic screen does not test a learned router, a neural Routing Network, natural task distributions, near-convergence capacity, or robust deployment latency. Fresh seeds were not opened by the frozen rule.
