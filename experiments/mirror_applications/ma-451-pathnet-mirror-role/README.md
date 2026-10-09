# MA-451 — PathNet path with Mirror module-role views

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-451-pathnet-mirror-role-20261008`  
Base commit: `1569029`  
Prior art: PA80 (PathNet)

## Hypothesis

H: Two adapted Mirror angles can let one reused PathNet path serve two distinct role orientations, improving held-out query quality over path-only and rank-2 control at a useful actual-byte/runtime point. A direct native Givens implementation can falsify Mirror-specific attribution.

**Mirror insertion:** this experiment adds two support-adapted role angles to the output of a selected PathNet module path so the shared path can implement two logical role functions without allocating duplicate modules.

## Task and controls

Two module choices per layer make four PathNet paths. For each path, two distinct angle-role tasks form an eight-task bank. Models meta-train shared module weights on synthetic episodes and adapt role coordinates on support examples. PathNet path-only (no role coordinate) is mandatory; additional controls are rank-2 additive residual per path, native Givens conditioning, and independent support-fit vectors. The task is a small linear regression mechanism screen, not a reproduction of the full PathNet paper.

Two development seeds (45101, 45102), 12 support examples, 40 query examples and four inner updates are frozen in `PROTOCOL.json`. Fresh seeds 45111–45113 stay sealed unless every development gate passes. Every path index, module, view code, residual basis and task vector is charged in an actual uncompressed `.npz` payload.

## H — Hypothesis

The frozen hypothesis is that two role angles let a reused PathNet route implement distinct functions, reducing query RMSE by 20% over path-only within a 1.25× payload allowance, while matching rank-2 adapter time. Native Givens output/payload equality negates Mirror-specific attribution.

## T — What ran

Two development seeds (45101, 45102). Each meta-trains for 160 outer updates over 8 sampled path/role tasks; adaptive methods see 12 support and 40 query examples with four inner updates. Path-only reads query examples only; adaptive conditions use support+query. An eight-function held-out bank enumerates four two-layer routes × two role orientations. Controls: mandatory PathNet path-only, Mirror role views, rank-2 additive residual per path, direct native Givens, and independent least-squares task vectors. Fresh seeds 45111–45113 remain sealed. Every route, module, coordinate, basis, task vector and schema is included in the measured uncompressed `.npz` payloads and per-seed `RESULTS_CORE.csv`.

## D — Result: FAIL

| Method | Seed 45101 RMSE / bytes | Seed 45102 RMSE / bytes | End-to-end ms (45101 / 45102) |
|---|---:|---:|---:|
| PathNet path-only | 0.7919 / 1,231 | 0.6003 / 1,231 | 0.28 / 1.58 |
| Mirror module role | 0.3926 / 1,557 | 0.2776 / 1,557 | 18.33 / 12.44 |
| Native Givens | 0.3926 / 1,557 | 0.2776 / 1,557 | 15.02 / 29.81 |
| Rank-2 additive role | 0.7519 / 1,943 | 0.5701 / 1,943 | 3.16 / 3.82 |
| Independent least squares | 0.0241 / 795 | 0.0243 / 795 | 1.06 / 1.24 |

Mirror cuts path-only RMSE by 50%/54%, demonstrating role-specific behavior on this aligned synthetic task. It fails the byte gate: 1,557 bytes is 1.265× the PathNet-only 1,231-byte payload, over the 1.25× cap, and is 1.96× the independent task-vector bank. It uses fewer bytes and has lower RMSE than the rank-2 residual, but support adaptation takes 5.8×/3.3× its measured time. Most decisively, direct native Givens conditioning has the exact same task codes, 1,557-byte hash and query RMSE in both seeds. Independent least-squares fit is far more accurate at 795 bytes.

**Fact:** Mirror function/task payloads are byte-identical to the native Givens control and replay to identical RMSE (0 difference) in both seeds. Mirror beats path-only and rank-2 RMSE; it misses the frozen actual-byte and runtime gates. The independent control is learnable (RMSE ≤0.025). Fresh seeds were not accessed.

**Interpretation:** Path-selected shared modules can express additional role behavior through a two-angle transform in this synthetic setup, but the 326-byte incremental role state exceeds the path-only storage budget, is costly to adapt, and is an ordinary native Givens parameterization.

**Hypothesis:** Module-role Views may be useful when the quality gain justifies per-role state and an optimized transform kernel exists; no Mirror-specific Pareto gain or PathNet-scale transfer is established.

### Worker report

- **H:** A small role coordinate lets reused PathNet modules support distinct functions without module duplication.
- **T:** Frozen two-seed task screen with path-only (required), rank-2, native Givens and independent-fit controls; actual payload and query replay verified.
- **D:** **FAIL** (quality improved, but actual bytes and runtime gates missed; exact native alias).
- **C:** Role variation was constructed from Givens angles, so the ordinary native conditioner fully explains the gain. More physical modules or an independent task vector achieve stronger quality.
- **U:** Natural PathNet tasks, trained neural modules, unseen paths/combinations, fused kernels, larger banks.

## Verification

`source/verify.py` validates every saved payload size/hash, replays query RMSE from serialized arrays, confirms native alias and checks that fresh seeds remain unopened. Four focused tests passed. No fresh data were opened.
