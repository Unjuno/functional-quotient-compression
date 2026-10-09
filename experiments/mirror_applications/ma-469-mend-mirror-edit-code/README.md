# MA-469 — MEND generates Mirror edit codes

Status: SCREENING (protocol/source frozen; development pending)  
Evidence lane: MECHANISM/STORAGE/EDIT_SUCCESS/LOCALITY/RUNTIME  
Branch: `research/ma-469-mend-mirror-edit-code-20261008`  
Base commit: `b92524d`  
Prior art: PA86 (MEND), PA87 (ROME)

## H — Hypothesis

A low-rank support-gradient signal can be mapped to two shared edit coordinates and decoded into useful per-edit weight changes with lower actual bytes than a MEND-style full-update editor, while preserving heldout edit success and locality. Direct native low-rank and analytic ROME controls isolate the contribution of the Mirror code.

## T — Frozen setup

A frozen 6×6 linear model receives 20 sequential edit requests. Each request has 24 support examples and 128 edit-query examples; locality uses 256 unrelated inputs. The target update lies in a seeded shared two-matrix subspace. Sixteen requests train the editor; four heldout requests are generated from support gradients only. Models train 2,200 Adam updates, batch size eight. Development seeds: 46901 and 46902; fresh seeds 46911–46913 remain sealed. The complete edit-system payload includes the base, editor, persistent per-edit update/code state, and key vectors. The operation proxy charges a 6×6 model application plus the editor input/output linear maps and (for Mirror) the shared-basis decode; query wall time and edit-generation wall time are timed separately.

## Prior-art delta

PA86's MEND maps low-rank edit-gradient decompositions into parameter updates. MA-469 compares direct full-matrix generation to compact code generation over a shared update basis. PA87's analytic rank-one update and an independent support-fit upper bound are included.

## Results

Pending frozen development runs.

## Development result

**Decision: NOT ESTABLISHED.** Both frozen development seeds violate the independent-fit locality validity bound; fresh seeds remain sealed. Mirror itself misses edit and locality thresholds on both seeds and is byte/output-identical to direct native low-rank generation.

### Facts

| Seed | Method | Edit RMSE | Locality RMSE | Payload bytes | Edit MAC proxy / edit |
|---:|---|---:|---:|---:|---:|
| 46901 | MEND | 0.10309 | 0.27438 | 15,192 | 2,340 |
| 46901 | Mirror | 0.10057 | 0.27151 | 8,526 | 1,324 |
| 46901 | Native low-rank | 0.10057 | 0.27151 | 8,526 | 1,324 |
| 46901 | ROME | 0.05486 | 0.09058 | 4,651 | 36 |
| 46901 | Independent fit | 0.00430 | 0.26609 | 4,670 | 36 |
| 46901 | No edit | 0.11172 | 0.00000 | 1,510 | 36 |
| 46902 | MEND | 0.05965 | 0.25099 | 15,192 | 2,340 |
| 46902 | Mirror | 0.07238 | 0.31140 | 8,526 | 1,324 |
| 46902 | Native low-rank | 0.07238 | 0.31140 | 8,526 | 1,324 |
| 46902 | ROME | 0.04618 | 0.09706 | 4,651 | 36 |
| 46902 | Independent fit | 0.00434 | 0.23267 | 4,670 | 36 |
| 46902 | No edit | 0.10526 | 0.00000 | 1,510 | 36 |

Training and generation wall time varied at sub-millisecond scale and is retained verbatim in `RESULTS_CORE.csv` and per-seed metrics; it is not treated as a stable runtime comparison. The initial preflight launch failure is recorded in `attempts.jsonl`; it emitted no experiment data.

### T — Execution

Two development worlds (46901, 46902), 2,200 Adam updates for each learned editor, exact NPZ inference payload byte counts, ROME and independent support-fit controls, serialized metric replay and native-alias checks. Seeds 46911–46913 were never opened.

### D — Decision

The frozen NOT ESTABLISHED rule fired: the independent support-fit control had edit RMSE 0.00430/0.00434 but locality RMSE 0.26609/0.23267, above 0.05. The development protocol therefore cannot validate a local-edit quality frontier. Independently, Mirror misses its own <=0.05 edit and locality gates and exactly matches the native low-rank control.

### C — Strongest counter-hypothesis

The synthetic task applies every edit as a global matrix update, while scoring locality on unrelated inputs. This mismatch may explain the upper-control locality failure; the outcome does not test realistic key-conditioned locality.

### U — Unconfirmed

Local/key-conditioned editing, factual language-model edits, near-converged MEND controls, fresh-world replication, and any storage/quality Pareto claim are unconfirmed.
