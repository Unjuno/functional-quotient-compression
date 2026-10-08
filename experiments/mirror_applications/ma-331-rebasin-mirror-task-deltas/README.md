# MA-331 — Re-Basin-aligned Mirror task deltas

Status: FAIL for the preregistered Mirror-specific storage margin at development; fresh seeds remain sealed. Dedicated branch: `research/ma-331-rebasin-mirror-task-deltas-20261008`.

## H — hypothesis

Permutation alignment may expose a compact two-direction task-delta orbit, and a single phase per aligned task may use at least 10% fewer actual bytes than a matched direct two-coefficient code while preserving function outputs.

## T — execution

Synthetic 2-layer ReLU MLP with 8 inputs, 12 hidden units and 4 outputs. Eight task functions share a base: six task deltas lie on a planted rank-2 circular output-head orbit, while two unrelated tasks use private deltas. The six aligned task MLPs are each independently permuted in hidden units. A Hungarian assignment on first-layer weights recovers the canonical hidden ordering. The inference payloads include shared base weights, delta basis, task codes and private unrelated deltas; alignment indices are not required after canonicalized weights are serialized.

Controls: independent full task MLPs; unaligned rank-2 task-vector compression; Re-Basin plus direct two-coefficient codes; Re-Basin plus one Mirror phase. Development seeds 33101/33102. No optimizer updates; this is an oracle-generated post-fit function screen, not task learning. Fresh seeds 33111–33113 were not accessed.

## D — decision

**FAIL for the preregistered Mirror-specific storage margin at development.** Both direct coefficients and Mirror phase preserve function outputs with nMSE below 1e-8. Mirror payload is 2,562B versus 2,574B direct: 12B (0.47%) smaller, short of the frozen 10% success gate. Payload shape is seed-invariant, so fresh cannot change this fixed byte comparison; fresh remains sealed.

The alignment mechanism itself is useful in this planted screen: unaligned rank-2 deltas have mean nMSE 0.20 and 0.264, while aligned direct and Mirror are below 1e-8. Independent full models use 4,934B. The strong result is from symmetry alignment and low-rank task deltas, not a Mirror-specific improvement over direct codes.

## C — strongest counter-hypothesis

A single phase is merely a compact way to write two ordinary coefficients on a known circular orbit. The direct control reconstructs the same functions, and the saved 12B is too small to meet the declared margin. The meaningful change is Re-Basin alignment, which reveals that the task deltas share a low-rank basis.

## U — unresolved

The screen uses planted synthetic task functions and oracle coordinates; no learned pretrained task deltas, natural data, interference behavior, or task adaptation cost were tested. The fresh split is sealed because the fixed development payload structures already fail the preregistered byte gate.

## Fact / interpretation / hypothesis

**Fact:** Eight development payloads were byte/hash checked and their FP16-reloaded task-output metrics replayed exactly. Four tests pass. Two seeds had identical payload sizes: 4,934B independent; 3,026B unaligned rank-2; 2,574B Re-Basin/direct; 2,562B Re-Basin/Mirror.

**Interpretation:** Re-Basin alignment turns permutation-scrambled task deltas into a compressible shared coordinate system. This Mirror phase did not add a useful storage/quality margin beyond the direct coefficient representation.

**Hypothesis:** Larger task banks or non-circular naturally learned deltas may have a different code frontier, but this run gives no evidence for that.

## Evidence files

- `RESULTS_CORE.csv`: development summaries.
- `artifacts/development/`: serialized inference payloads and metrics.
- `source/verify.py`: byte/hash and reloaded metric replay.
- `VERIFICATION.json`: verification record; fresh integrity intentionally false because fresh data stayed sealed.
