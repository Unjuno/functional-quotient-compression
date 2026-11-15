# MA-1037 results — FAIL

## H — Hypothesis

At fixed Matryoshka prefix dimension, one train-qrels-selected Givens coordinate at the prefix boundary could improve held-out SciFact retrieval quality over the native prefix at nearly the same complete inference bytes and acceptable query-to-top-10 latency. The gain would need to exceed a normalized two-coordinate linear boundary readout to support Mirror-specific value.

## T — Test

- **Model:** Nomic Embed v1.5, pinned revision `e9b6763023c676ca8431644204f50c2b100d9aab`, official INT8 ONNX export; CPU ONNX Runtime 1.22.0. Model-card prompts, masked mean pooling, full-dimension layer norm, and L2 normalization.
- **Data:** BEIR SciFact, 5,183 corpus documents. Development used 809 train-qrels query IDs; fresh audit used 300 disjoint test query IDs and all 5,183 corpus documents. Exact full-corpus cosine retrieval, nDCG@10, Recall@10, and MRR@10.
- **Views:** dimensions 64, 128, 256, 512, 768. One boundary angle per reduced dimension was selected from a frozen 21-value grid on train qrels only: d64 −π/2, d128 0.94248, d256 1.25664, d512 −0.15708; d768 fixed to zero. Matched native-prefix, normalized two-coefficient boundary readout, and full 768-D controls.
- **Compute:** embedding all corpus and query text took 1,354.9 s and 20.4 s respectively on CPU (3.83 docs/s and 54.38 queries/s). All dimension arms share these embeddings. Retrieval latency includes query readout, exact score matrix, top-10 selection, and ordering. The first FP32 encoder attempt and audit execution/timing/manifest corrections are disclosed in `AMENDMENT_01.json`, `POST_DEV_AMENDMENT.md`, and `POST_AUDIT_AMENDMENT_02.md`; settings and gates were not retuned.

## D — Decision: FAIL

### Facts

| Dimension | Native nDCG@10 | Mirror nDCG@10 | Difference | Native Recall@10 | Mirror Recall@10 | Native QPS | Mirror QPS | Native complete bytes | Mirror complete bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.50992 | 0.51307 | +0.315 pp | 0.62583 | 0.62750 | 35,090 | 34,393 | 155,548,431 | 155,548,488 |
| 128 | 0.59885 | 0.59857 | −0.028 pp | 0.72772 | 0.73372 | 29,498 | 30,823 | 156,875,281 | 156,875,338 |
| 256 | 0.65742 | 0.65692 | −0.050 pp | 0.78928 | 0.78761 | 26,256 | 24,616 | 159,528,977 | 159,529,034 |
| 512 | 0.68343 | 0.68348 | +0.005 pp | 0.80844 | 0.80844 | 20,370 | 16,491 | 164,836,369 | 164,836,428 |
| 768 | 0.68992 | 0.68992 | 0.000 pp | 0.82122 | 0.82122 | 16,455 | 16,245 | 154,221,455 | 154,221,455 |

The best held-out nDCG gain is 0.315 percentage points at 64 dimensions, below the 0.5-point gate. No sub-full dimension reaches the required gain; the other three are effectively tied or slightly worse. The 768-D zero-view exactly reuses the full source table and adds no candidate bytes.

The matched normalized linear boundary control is bit-exact to the Givens query readouts and candidate tables at all five dimensions. It has identical retrieval quality. This directly fails Mirror-specific attribution.

The complete materialized five-budget bank is 174,126,309 B for native prefixes and 174,126,539 B for Mirror, a 230 B increase. This includes the 138,243,620 B model/tokenizer bundle, 15,922,304 B full source vector table, every candidate table, corpus IDs, coordinates, and manifests. The extra four float32 angles total 16 B. On the final manifest-complete run, d512 query-to-top-10 throughput is 19.0% lower than native. A preceding run with identical query timing produced materially different QPS ordering across dimensions, although all quality metrics were bit-identical. Runtime is therefore NOT ESTABLISHED; the repeated short 300-query batches and shared-host load do not support a stable throughput conclusion.

Exact payload replay reproduces all 15 method×dimension byte counts and all 45 quality metrics with zero difference. Five tests pass. Thirty-three serialized payload artifacts are hashed. Audit query IDs do not overlap development IDs.

### Interpretation

The boundary Givens coordinate can slightly reshape a low-dimensional Matryoshka readout, but the observed quality movement is too small to extend the useful quality-cost frontier. The matched two-weight linear readout produces exactly the same function, so any observed movement is ordinary boundary projection freedom rather than a Mirror-specific benefit. The full 768-D orthogonal view changes no retrieval function.

The full source embedding table is charged alongside each materialized candidate index. That makes the small-angle coordinate cheap, but the experiment does not compress the encoder or the multi-budget index bank. Encoding itself remains the dominant compute cost and is unchanged across dimensions.

## C — Strongest counter-hypothesis

Matryoshka training already places most useful retrieval signal in its native prefixes. The only new degree of freedom tested is one two-coordinate boundary projection, which is exactly representable by ordinary learned linear weights. Small score changes are likely benchmark/grid-selection variation rather than a distinct functional compression mechanism.

## U — Unconfirmed

This is one scientific retrieval dataset and one quantized pretrained encoder, with exact dense retrieval rather than ANN indexing. No encoder fine-tuning, second task, natural-language domain shift, GPU inference, or raw FP32 retrieval comparison was run. The CPU embedding cost is high, and the audit does not support claims about other Matryoshka models or learned projections with larger rank.

## Verification

Five geometry/metric/serialization tests pass. `source/verify_results.py` reloads actual `.npy` payloads and reproduces exact bytes and retrieval metrics. See `VERIFICATION.json`, `runs/VERIFICATION_REPLAY.json`, and `runs/PAYLOAD_HASHES.json`.


The final manifests contain resolvable relative paths for the shared source vectors, candidate tables, corpus ID map, optional coordinates, and model artifact. The pre-correction audit outputs are retained in `runs/AUDIT_RESULTS_PRE_MANIFEST_FIX.json`.
