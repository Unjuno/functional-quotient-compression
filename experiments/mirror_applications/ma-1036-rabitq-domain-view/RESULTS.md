# MA-1036 results — FAIL

## H — Hypothesis

A shared paid FAISS random rotation plus one compact per-domain Givens coordinate could approach four independently rotated RaBitQ indexes, while using at least 25% fewer complete inference bytes and improving at least three of four domains by 2 Recall@10 percentage points over one shared rotated index. A post-quantization orthogonal gauge was expected to leave rankings unchanged.

## T — Test

FAISS `IndexRaBitQ` 1.16.0, CPU, D=128, 1 bit/dimension, 32 synthetic clusters. Each world has 2,048 training vectors, one 2,048-vector database, and 256 queries in each of four covariance domains. Exact top-10 is Euclidean L2. Development world 103600 selected one angle/domain from the frozen eight-angle grid: `[0, 2.74889, 1.96350, 0.39270]` radians. These values were locked for fresh worlds 103601–103603.

Controls: unrotated single stream, shared random-rotation single stream, shared basis + four domain-specific views and code streams, and four independent random rotations/code streams. All serialized indexes, matrices, angle metadata, and manifests are included in byte counts. Serialized indexes were reloaded for exact payload and Recall@10 replay. A separate post-quantization gauge rotates queries and decoded reconstructions together.

## D — Decision: FAIL

### Facts

| Method | Mean Recall@10 | Mean Recall@1 | Actual inference bytes | Mean FAISS train+add wall |
|---|---:|---:|---:|---:|
| Unrotated shared stream | 3.024% | 4.980% | 49,891 | 0.0040 s |
| Shared random rotation | 2.930% | 5.241% | 115,424 | 0.0042 s |
| Shared basis + domain views | 2.799% | 4.655% | 264,683 | 0.0216 s |
| Independent rotations | 2.933% | 5.046% | 461,251 | 0.0161 s |

The view bank uses 42.6% fewer bytes than the independent rotation bank, meeting the storage gate. Its mean Recall@10 is 0.13 percentage points below the independent bank and therefore within the 1-point quality tolerance. However, it does not improve the shared-index baseline by 2 points in any domain. Mean domain-wise Mirror-minus-shared differences are 0.00, -0.27, +0.16, and -0.40 percentage points. The main quality and usefulness gates fail.

Mean absolute squared-distance estimator errors on returned top-10 entries are 34.153 for Mirror and 33.838 for independent rotation. Mean signed errors are -32.120 and -31.727, respectively. The tested estimator is biased low in every arm.

Post-quantization gauge rankings were identical in all three fresh worlds (zero changed neighbor IDs), using the same code stream plus a paid 4-byte float angle. No additional ranking function was observed.

Transform+search QPS was not stable enough to decide its gate: three runs of the same frozen fresh configuration produced Mirror/independent mean QPS pairs of 25,081/28,660, 29,202/22,513, and 28,077/38,563. This is disclosed as NOT ESTABLISHED; no favorable run was selected. Index build times in the table measure FAISS train+add for each unique physical index only. See `POST_AUDIT_AMENDMENT.md` for the two transparent instrumentation corrections and retained initial outputs.

### Interpretation

A small per-domain angle can alter RaBitQ sign-hyperplane quantization and approach independent-rotation recall with fewer bytes in this synthetic setting. But ordinary shared rotation already matches the independent bank, while domain views add almost 150 kB and fail to improve it. The bytes saved versus independent rotations do not translate into a useful storage-quality frontier against the simpler shared-index control.

The post-quantization result is an isometric gauge invariance check, not extra capacity. The very low absolute Recall@10 and synthetic distribution limit applicability.

## C — Strongest counter-hypothesis

Any apparent benefit is distribution/rotation tuning for a one-bit quantizer rather than Mirror-specific logical multiplicity. The shared random rotation already captures the available performance here; the selected per-domain angles neither reliably improve quality nor beat the simpler shared stream on the complete storage frontier.

## U — Unconfirmed

No SIFT, BigANN, natural embedding, higher-bit RaBitQ, learned dense rotation, or production CPU workload was tested. Runtime remains not established because short 256-query batches produced materially different throughput on repeated runs. This screen does not characterize other quantizers or broader domain adaptation.

## Verification

Four unit tests pass. Twelve method×world payloads reproduce their exact serialized byte counts and Recall@10 after loading the actual FAISS index files; 60 payload file hashes are recorded. Audit seeds are disjoint from the development seed. Full details are in `runs/`.
