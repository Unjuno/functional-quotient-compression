# MA-962 results — task × time-step TEBN Mirror View

## H — Hypothesis

A rank-4 task×time normalization code can preserve full per-task TEBN quality with lower actual serialized bytes and comparable dynamic operation cost when temporal signatures have low rank. Generic signatures test where private/full state becomes necessary. Rank-4 Mirror is compared with the exact ordinary rank-4 factorization control.

## T — Test

**Fact.** CPU custom PyTorch SNN with 16 event bins, 8 Bernoulli input channels, fixed shared 8→2 encoder, shared BN running statistics with affine scale/shift frozen, and two LIF cells. LIF uses leak .9, threshold 1, hard spikes with triangular surrogate gradient, and subtractive reset. Eight tasks; positive/negative event rates encode task-specific temporal signature `K[q,t]`. Adam .02, batch 128, 1,000 updates, one CPU thread. Fresh structured seeds 96211, 96213, 96217; corrected generic seeds 96241, 96243, 96247. Each world uses 256 train and 512 test examples/task. For generic-random, train/test use the same task-signature matrix (`bank_seed`) and independent sample RNG seeds.

Compared shared TEBN gain, full task×time gain table, task-scalar×time rank-1, rank-4 Mirror factors, and identical native rank-4 factors. Rank-4 conditions used the same initialization, training batches, optimizer, and equations; all state tensors and row metrics were asserted equal.

## Results

**Fact.** Means across three fresh worlds per task family:

| Task bank | Condition | Accuracy | Payload bytes | Mean spikes/example | First-correct latency (steps) | Inference examples/s |
|---|---|---:|---:|---:|---:|---:|
| Structured rank-3 | Shared TEBN | 0.99943 | 720 | 7.149 | 4.438 | 45,333 |
| Structured rank-3 | Full task×time TEBN | 0.99992 | 1,168 | 8.142 | 3.149 | 66,358 |
| Structured rank-3 | Rank-1 factor | 0.50000 | 824 | 0.000 | 16.000 (censored) | 57,511 |
| Structured rank-3 | Rank-4 Mirror / native | 1.00000 | 1,119 | 9.039 | 4.158 | 73,453 |
| Generic random | Shared TEBN | 0.92757 | 718 | 5.200 | 4.874 | 54,921 |
| Generic random | Full task×time TEBN | 1.00000 | 1,166 | 9.207 | 3.052 | 62,748 |
| Generic random | Rank-1 factor | 0.96899 | 822 | 8.037 | 5.241 | 41,484 |
| Generic random | Rank-4 Mirror / native | 0.99976 | 1,117 | 11.522 | 2.291 | 74,402 |

Rank-4 uses 49 B fewer than full (4.2%); it misses the preregistered 20% byte reduction. Rank-4 mean spikes exceed full by 11.0% on structured and 25.1% on generic random, beyond the 10% dynamic-operation gate. Structured first-correct latency is 32.0% slower than full; generic random is 24.9% faster. Thus the structured latency gate fails, while generic latency passes. Throughput varied greatly across seeds and between the exact rank-4 duplicates, so it is not treated as stable evidence.

**Fact.** On structured data, task-agnostic shared TEBN already reached 99.94% with 720 B and fewer spikes than rank-4. On generic-random tasks, shared TEBN reached 92.76%; rank-4 improves quality to 99.98%, but costs 399 B more than shared TEBN and emits 121% more spikes. The full table reaches 100% at 1,166 B.

**Fact.** Serialized reload replay max absolute error was 0.0. Mirror and native rank-4 state tensors, per-task accuracy and payloads were identical. The initially reported generic-random development failure was invalid: train and test had different K matrices. That result is preserved in `source/development_invalid_split.json` and withdrawn. After fixing task-signature identity, corrected development full TEBN reached 99.98%; fresh generic evaluation used new seeds 96241/96243/96247. An initial fresh process on retired seeds 96231/96233/96237 was interrupted before a result row and excluded; see `source/generic_interrupted.json`.

**Interpretation.** Rank-4 factors preserve quality on both tested signatures, but yield only a small actual-byte saving versus full TEBN while adding spike activity. On structured data, the simpler shared TEBN is already sufficient. On generic tasks, task conditioning helps quality over shared TEBN, but rank-4 still fails the byte and spike-count gates. The rank-4 implementation is ordinary low-rank temporal factorization, not Mirror-specific value.

## D — Decision

**FAIL (scoped structured and generic screens).** Both banks miss the required actual-byte improvement; both exceed the spike-count regression gate. The generic task bank now has a valid native full-table baseline. No TEBN paper reproduction, natural event result, capacity increase, or energy benefit is established.

## C — Strongest counter-hypothesis

The result may reflect a small task bank and the serializer's fixed manifest cost, which limits amortization. Larger task/time banks may reduce manifest overhead and make low-rank state more useful. This would not change the exact alias with ordinary low-rank factorization.

## U — Unconfirmed

No natural event stream, official TEBN benchmark/code, learned synapse network, GPU, hardware power, or joules were measured. Task IDs were available to all conditioned methods and charged in the payload. Generalization to unseen tasks or temporal signatures remains untested.

## Verification and data correction

The fresh command completed for all six seed×bank combinations. The common serializer was reloaded and evaluated; max replay difference was 0.0. Rank-4 Mirror/native state tensors were asserted equal. A post-run audit found the first generic-random train/test split generated different K matrices; that evidence was withdrawn before corrected generic fresh evaluation. The corrected run locks one K per world and uses separate sample seeds. A premature process on three retired seeds was interrupted and excluded.
