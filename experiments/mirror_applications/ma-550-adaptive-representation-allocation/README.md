# MA-550 — Adaptive representation allocation

Status: **FAIL for Mirror-specific attribution; synthetic allocation mechanism passes**
Branch: `research/ma-550-adaptive-representation-allocation-20261009`
Prior art: PA96 (ReFT / LoReFT)
Base checkpoint: pinned Pythia-70M-deduped, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`

## H — Hypothesis

A per-function selector can allocate synthetic output changes between activation coordinates and sparse/dense output-weight codes with fewer actual serialized bytes than either fixed allocation while meeting a frozen reconstruction tolerance. A native sparse-bias plus shared output-basis/ReFT control may exactly explain any gain.

## T — Execution

Used the pinned output projection `W` (vocabulary 50,304 × width 512). Each world contains 16 functions: eight single-token logit shifts and eight dense shifts generated as `Wm`, with `m ~ N(0, .05² I)`. Each function has eight noisy support observations and eight independent held-out observations. The two development seeds were 55001/55002; fresh seeds 55011/55013 were opened only after development decision commit. No optimizer updates or model training were used.

Compared fixed dense output deltas, fixed activation codes solved through the already-paid `W`, fixed one-coordinate sparse output biases, and an adaptive byte-minimizing selector over the same native methods. Actual uncompressed NPZ inference payload bytes include tags, IDs, seed/model metadata and values. The 168,144,624 B base checkpoint is counted once for full-system context. All per-world NPZ payloads, metrics, hashes, and split-free deterministic generated worlds are retained under `results/`.

## D — Decision

**FAIL for Mirror-specific value.** The synthetic heterogeneous allocation gate passes in all three fresh worlds: adaptive payload is 19,682 B, support relative RMSE is at most 0.00398, and held-out relative RMSE at most 0.000105. It is 163.6× smaller than fixed dense output deltas. It uses 42.9% fewer incremental bytes than fixed activation codes, but that fixed activation baseline fails quality on the sparse functions (held-out relative RMSE about 0.497).

The selected allocation is always eight ordinary activation codes plus eight ordinary sparse output-bias entries. This is exactly the native shared output basis/ReFT plus sparse-bias mixture and a conventional support-fit selector. It adds no Mirror-specific functionality. The activation decodes require an estimated 412,090,376 FLOPs per 16-function bank, versus 804,864 dense-vector additions for the all-weight delta bank; the adaptive point trades compute for bytes. Since the common model is 168.1 MB, total model-plus-payload savings versus dense deltas are 3,201,160 B (1.87%).

## H / T / D / C / U

**H:** Frozen above; heterogeneous output changes should favor different representation families.
**T:** Two development and three fresh worlds, pinned Pythia output matrix, 8 support / 8 held-out observations, four serialized controls, no optimizer.
**D:** FAIL for Mirror-specific attribution; all fresh worlds pass the synthetic mixed-allocation quality/bytes gate.
**C:** Native sparse output biases plus ReFT-style coordinates through the shared output basis and a standard support-fit selector give the same encoding choices and quality.
**U:** Natural knowledge edits, learned task routing, other layers/models, end-to-end latency, energy, and function distributions outside this synthetic mixture are untested.

## Fact / Interpretation / Hypothesis

**FACT:** All five worlds select 8 activation codes and 8 sparse-bias codes; adaptive uses 19,682 B with fresh held-out relative RMSE 2.44e-5–3.21e-5 (world mean), and every fresh maximum per-function error is ≤1.05e-4. Fixed dense deltas use 3,220,842 B and remain under 0.004 per-function error. Fixed all-activation and all-sparse controls fail because each cannot express the other family. Development 55001 replay reproduced all four NPZ payloads byte-for-byte.

**INTERPRETATION:** Representation heterogeneity can reduce bytes on this deliberately mixed linear output family; this is an ordinary representation-selection result. The compute/storage point improves over dense weights on bytes and over fixed activation on both bytes and compute, but the all-activation quality miss and synthetic setup limit the frontier claim.

**HYPOTHESIS:** Natural tasks may also contain edits with different cheapest sufficient representations, but a Mirror-specific advantage requires beating the native sparse-bias/ReFT selector at equal bytes, compute and quality.
