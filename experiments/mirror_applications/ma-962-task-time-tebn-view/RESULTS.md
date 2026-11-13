# MA-962 results — task × time-step TEBN Mirror View

## H — Hypothesis

A compact rank-4 task×time normalization code can preserve a full per-task TEBN gain table with lower actual serialized bytes and comparable dynamic operation cost when temporal signatures have low rank. Generic signatures test where private/full state becomes necessary. Rank-4 Mirror is also compared with the exact ordinary rank-4 factorization control.

## T — Test

**Fact.** CPU custom PyTorch SNN with 16 event bins, 8 Bernoulli input channels, fixed shared 8→2 encoder, shared BN running statistics with affine scale/shift frozen, and two LIF cells. LIF uses leak .9, threshold 1, hard spikes with triangular surrogate gradient, and subtractive reset. Eight tasks; positive/negative event rates encode task-specific temporal signature `K[q,t]`. Adam .02, batch 128, 1,000 updates, one CPU thread. Fresh seeds 96211, 96213, 96217; 256 train and 512 test examples/task. The declared inference serializer is one typed tensor stream plus JSON manifest, and includes encoder, BN state, gain state, task metadata, and LIF constants. Loaded serialized state was used for evaluation.

Compared shared TEBN gain, full task×time gain table, task-scalar×time rank-1, rank-4 Mirror factors, and identical native rank-4 factors. Rank-4 conditions used the same initialization, training batches, optimizer, and equations; all state tensors and row metrics were asserted equal.

The generic-random bank was **NOT ESTABLISHED** at development: full native TEBN reached 46.5% after 1,000 updates and 47.3% after a 5,000-update development extension, below the 90% native feasibility gate. No generic-random fresh scores are reported. The exact event definition and amendment history are in `PROTOCOL.json`; development evidence is preserved under `source/`.

## Results

**Fact.** On the fresh structured rank-3 temporal bank, averages across three worlds:

| Condition | Accuracy | Payload bytes | Mean spikes/example | First correct latency (steps) | Inference examples/s |
|---|---:|---:|---:|---:|---:|
| Shared TEBN | 0.99943 | 720 | 7.149 | 4.438 | 45,333 |
| Full task×time TEBN | 0.99992 | 1,168 | 8.142 | 3.149 | 66,358 |
| Task-scalar×time rank-1 | 0.50000 | 824 | 0.000 | 16.000 (censored) | 57,511 |
| Rank-4 Mirror | 1.00000 | 1,119 | 9.039 | 4.158 | 67,956 |
| Ordinary rank-4 factorization | 1.00000 | 1,119 | 9.039 | 4.158 | 78,949 |

Rank-4 reduced the full-table payload by 49 B (4.2%), below the preregistered 20% saving gate. It added 64 gain-factor multiplies/example relative to direct full-table row lookup. Its mean spike count was 11.0% above full TEBN and first-correct latency 32.0% slower, missing both dynamic-operation gates. Shared TEBN, with no task-conditioned state, was 99.94% accurate in all three fresh worlds, had the smallest payload, and emitted fewer spikes than rank-4.

**Fact.** Serialized replay maximum absolute difference was 0.0. Mirror and native rank-4 tensor state, accuracy per task, and serialized payload were identical. Throughput varied substantially between seeds and is not treated as a stable speed result.

**Interpretation.** In this structured synthetic case, task-conditioned rank-4 factors retained quality but did not provide a useful Pareto improvement: bytes barely fell against the full table, dynamic behavior worsened, and the cheaper task-agnostic shared TEBN already solved the task. The rank-4 implementation is an ordinary low-rank temporal factorization, not Mirror-specific value.

## D — Decision

**FAIL (scoped structured screen); generic-random bank NOT ESTABLISHED.** The structured case missed actual-byte and spike/latency gates. No TEBN paper reproduction, natural event result, capacity increase, or energy benefit is established.

## C — Strongest counter-hypothesis

The structured teacher and one shared scalar task were unusually easy: task-agnostic gain already recovered the labels. A different temporal task family may require task-conditioned gains and produce a larger storage frontier. The generic-random native full control did not learn adequately, so it cannot adjudicate that case.

## U — Unconfirmed

No natural event stream, official TEBN benchmark/code, learned synapse network, GPU, hardware power, or joules were measured. Task IDs were available to all conditioned methods and charged in the payload. A useful result on task-temporal functions beyond this synthetic signature remains untested.

## Verification

The fresh command completed for all three seeds. The common serializer round-tripped all model tensors and loaded states for inference with exact logit replay (maximum absolute difference 0.0). The rank-4 Mirror/native factor state tensors and row metrics were exactly equal by assertion.
