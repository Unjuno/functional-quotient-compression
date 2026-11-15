# MA-1042 — PGM-index domain residual View

Status: **FAIL for Mirror-specific value**. The experiment demonstrates an affine-domain shared-index compression mechanism on synthetic data, but the matched ordinary affine-coordinate control reproduces the same exact functions. The registered Mirror-specific gate therefore fails.

## H — hypothesis

A compact per-domain coordinate can map four affine-related ordered key domains to one physical PGM index while preserving exact predecessor answers and insert/delete outcomes. The shared complete serialized payload should be at most half the independent-index payload, and p95 lookup latency should stay within 1.10×. A Mirror-specific result must also improve on the matched native affine-coordinate control.

## T — execution

- Official gvinciguerra/PGM-index, Apache-2.0, pinned source commit c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407; g++17 CPU implementation.
- 16,384 strictly increasing generated canonical keys; four affine domains (scale, offset) = (3,1e9), (5,4e9), (7,9e9), (11,16e9); each world applies 128 inserts and 128 deletes.
- Development seeds 1042/1043 chose epsilon 8 under the frozen public DynamicPGMIndex RAM rule. Confirmatory fresh seeds 1050/1051/1052 each used 10,000 queries per domain (40,000 total), five interleaved latency passes, and exact-vector predecessor controls.
- Controls: one shared index plus domain coordinates, matched ordinary native affine coordinate, four independently stored/indexed domains, and a no-view shared-index negative control.
- The official DynamicPGMIndex applies updates; the official static PGMIndex is rebuilt on post-update keys for predecessor queries because the pinned dynamic iterator has no predecessor-decrement interface. Dynamic membership, rebuild cost, and query correctness are reported separately.
- AM-03 corrected left-side predecessor refinement after an initial held-out implementation defect; seeds 1044–1046 remain preserved and excluded. AM-04 corrected premature byte measurement before shared-config flush; seeds 1047–1049 remain preserved and excluded. Only 1050–1052 are confirmatory for the final implementation.

## Facts

- All 120,000 confirmatory predecessor answers were exact for Mirror, affine control, and independent indexes; all insert/delete membership assertions passed. The no-view control returned the exact predecessor in 0/40,000 cases.
- Actual on-disk payload, including keys, ordered update logs, all four domain codes, and configuration metadata: **133,720 B shared** vs **534,260 B independent**, a **74.97% reduction**. The reported byte counters match filesystem walks for all three worlds.
- Shared final-predictor RAM (final key array + static PGM) was 131,464–131,544 B vs 525,856–526,176 B independent. DynamicPGMIndex update-state RAM was 266,984 B shared vs 1,067,936 B independent.
- p95 lookup latency by fresh world was 0.000106/0.000105/0.000105 ms shared and 0.000110/0.000106/0.000106 ms independent; all ratios pass the 1.10× gate. This is a single-host CPU microbenchmark, not end-to-end database latency.
- Mean initial DynamicPGM build was 1.18 ms shared vs 5.89 ms independent; mean update time 0.213 vs 1.286 ms; final static rebuild 0.495 vs 1.845 ms.
- The Mirror-labelled and ordinary affine paths execute the same integer coordinate mapping and both returned exactly the same answers. Their small timing differences are measurement noise; no distinct Mirror operation is present.

## Interpretation

The synthetic affine redundancy supports sharing one physical ordered key set and predictor across four logical domains: it preserves exactness and cuts this serialized state by about three quarters while staying within the measured latency gate. That saving is available from ordinary affine coordinate handling and hard sharing. It is not evidence for a Mirror-specific capability or compression advantage. The dataset was constructed to have exact affine structure; arbitrary or independently distributed databases were not tested.

## D — decision

**FAIL** for Mirror-specific attribution, because the matched ordinary affine-coordinate control is functionally identical. The generic shared-index mechanism passes its bounded exactness, storage, and CPU p95 gates.

## C — strongest counter-hypothesis

The entire result is explained by lossless affine re-encoding of deliberately related key sets plus one shared canonical index. Ordinary database coordinate normalization achieves the same behavior, and the registered native affine control is an exact alias.

## U — unconfirmed

- Natural databases with partially shared or non-affine key distributions.
- Online predecessor serving directly from a dynamic index; this pinned dynamic API required a static predictor rebuild after updates.
- Concurrent updates, range-scan throughput, SSD behavior, tail latency under load, and larger-scale storage.
- Any Mirror-specific benefit over native coordinate normalization.

## Verification and provenance

Four payload/replay tests pass. The independent verifier reconstructs each domain's initial keys and updates from the shared state and checks recorded exactness, payload byte equality, and the frozen storage/runtime gates. See VERIFICATION.json, runs/DEV_FREEZE.json, and runs/audit/. Superseded audit data remain under runs/exploratory_invalidated/.
