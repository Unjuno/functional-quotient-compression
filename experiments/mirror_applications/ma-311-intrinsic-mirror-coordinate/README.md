# MA-311 — Mirror task codes in a random intrinsic subspace

Status: **FAIL for the preregistered all-world storage/view gate; scoped storage result retained**
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base: MA-307 published branch, `bfabb6e`
Prior art: PA33 (random intrinsic-subspace tuning is the required direct control).

## H — hypothesis

**Hypothesis:** after fitting task deltas in a frozen random intrinsic subspace, one shared intrinsic vector plus compact Givens Mirror coordinates represents aligned task functions with at least 15% fewer serialized bytes than ordinary intrinsic task vectors, while unrelated tasks use private intrinsic coordinates.

**Mirror insertion:** add a one-byte angular coordinate to one shared 64-dimensional intrinsic task vector. This represents orbit-aligned task deltas without storing a separate 64-value vector per task.

## T — execution

Synthetic linear task functions have input dimension 128 and eight tasks. The first task supplies a shared 128-value fitted base. Four subsequent tasks are generated on a Givens orbit within a deterministic random 64-dimensional orthonormal projection; three tasks have unrelated random intrinsic deltas. Methods: hard-tied base, ordinary random-subspace vectors (PA33), Mirror angle plus private intrinsic fallback, and independent full 128-dimensional vectors. Each task has 512 support, 256 validation, and 512 held-out examples. No gradient updates are used.

The original d=32 development screen missed the preregistered <=0.85 payload ratio (0.865); it is retained in `protocol_variants/d32_pre_amendment/`. Before any fresh access, the protocol was amended to test d=64. Development worlds 31120/31121 selected validation threshold 0.0001 from {0.0001, 0.001}. Fresh worlds 31122–31124 used the locked d=64 and threshold without retuning.

Actual serialized payload counts include the shared base, task codes/private coordinates, header, and deterministic projection seed/recipe metadata. The basis is regenerated exactly from that paid seed and recipe. `RESULTS_CORE.csv` has 28 method/world rows; `ALLOCATION_EVENTS.csv` records 224 task allocation events.

## D — decision

The strict all-world promotion gate **failed**. Fresh Mirror payloads were 1,384, 1,506, and 1,383 bytes versus 1,753 bytes for ordinary intrinsic vectors (ratios 0.790, 0.859, 0.789). Thus savings were 21.0%, 14.1%, and 21.1%; the middle world missed the 15% minimum. The locked validation criterion accepted all four aligned non-anchor tasks as views in two worlds; one aligned task used a private intrinsic fallback in the remaining world. All three unrelated tasks used private intrinsic coordinates in every world. Mirror held-out mean normalized MSE was 1.01e-5, 5.81e-6, and 1.93e-5; ordinary intrinsic control was about 1.6–1.9e-8. The strict quality threshold 1e-4 was met, but Mirror quality was worse than the direct intrinsic control.

Relative to independent full vectors, Mirror used about 31–34% of serialized bytes, with a quality gap. Its active-compute proxy was 23.90M operations versus 23.59M for ordinary intrinsic vectors; its fitted angle search adds work and observed decode/inference throughput was lower in these CPU diagnostics. This does not support an overall Pareto improvement.

## Fact / interpretation / hypothesis

- **Fact:** in 3/3 fresh worlds, Mirror payloads were 14.1–21.1% smaller than ordinary intrinsic-vector payloads. Two worlds encoded all aligned tasks as views; one required one extra private intrinsic vector. All unrelated tasks required private coordinates. Mirror held-out error was around 5.8e-6–1.9e-5 versus around 1.6e-8–1.9e-8 for ordinary intrinsic vectors.
- **Interpretation:** a compact coordinate can exploit a known orbit inside an intrinsic subspace, but it does not recover ordinary intrinsic fit accuracy, and its storage gain narrowly misses the strict all-world requirement. Unrelated functions still need private state.
- **Hypothesis:** a finer/multidimensional orbit address might improve the accuracy/byte tradeoff, but requires a new experiment and is not established here.

## C — strongest counter-hypothesis

The teacher tasks were deliberately generated from the same Givens-orbit family used by the Mirror code. Even on this favorable structure the ordinary intrinsic coefficient control is more accurate, and one of three fresh worlds misses the target compression ratio. A natural task stream may have less orbit alignment.

## U — unresolved

No pretrained backbone, language-model quality, learned task router, gradient-based optimization, near-convergence capacity, or natural task distribution was tested. Timing is a small CPU mechanism diagnostic, not a deployment benchmark.

Protocol: [PROTOCOL.json](PROTOCOL.json). Metrics: [RESULTS_CORE.csv](RESULTS_CORE.csv). Allocation trace: [ALLOCATION_EVENTS.csv](ALLOCATION_EVENTS.csv). Exact replay: [source/REPLAY_VERIFICATION.json](source/REPLAY_VERIFICATION.json).
