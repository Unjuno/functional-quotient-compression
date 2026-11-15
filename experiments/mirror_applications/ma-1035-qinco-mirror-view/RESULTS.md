# MA-1035 — QINCo residual-stage Views

Status: **FAIL** for the preregistered QINCo quality/runtime and Mirror-specific gates. This is a small synthetic CPU screen, not a large-scale ANN claim.

## H — Hypothesis

One shared official-structure QINCoStep, with a small per-stage Givens coordinate on its decoded-prefix condition, will preserve fresh-world Recall@10 while saving at least 25% actual model+index bytes against independent QINCo stages and outperforming a byte-near additive stage-control.

## T — Trial

- Pinned upstream Facebook QINCo repository commit `5a324954d5c9b3700d4407d6cc24c3db6e52890e`; original QINCo v1 model source SHA-256 is recorded in `ARTIFACT_PROVENANCE.json`. The copied native `QINCo` / `QINCoStep` definitions are used for the independent-stage control; only the helper import path is redirected to a local exact PyTorch nearest-assignment implementation.
- Small model: d=32, K=16, M=4, L=1, hidden=32. Training uses stagewise torch k-means residual initialization, 80 Adam updates, batch64 and lr .001.
- Four independent synthetic vector worlds, each with 16 newly sampled anisotropic Gaussian clusters: dev 103500; three fresh audit worlds 103501–103503. Each world contains 1,024 train vectors, 256 validation vectors, 512 database vectors and 128 queries. Ground truth is exact Euclidean top-10 against the uncompressed database.
- Controls: stage-specific static residual quantizer; official-structure independent-stage QINCo; hard-tied QINCoStep; shared QINCoStep plus Givens stage coordinates; shared QINCoStep plus rank-1 additive conditioning.
- Model payloads include all state tensors and self-describing manifests. Index files include packed 4-bit codes (two bytes/vector) and row-order metadata. Database vectors are reconstructed at load time from paid model/index state; 65,536 B of resident float32 reconstruction memory and decode wall are reported. Database encoding wall is the median of five runs; decode and complete 128-query exact scans are medians of 20 runs after warm-up.
- All three fresh worlds were independently generated and all five arms were trained/evaluated within each world. Four unit tests pass; full state/code/payload and metric replay passes exactly over 15 fresh method-world conditions.

## D — Decision

**FAIL.** Mirror total model+index payload averages 22,634 B vs native independent QINCo's 60,386 B, a 62.5% reduction that passes the storage gate. However, mean Recall@10 is 42.34% vs 47.81% native QINCo, a 5.47 percentage-point loss versus the allowed 2 points. Its load+scan throughput averages 93,725 queries/s vs 125,228 for native (25.2% lower), missing the 10% latency-regression gate. Against matched additive conditioning, Mirror improves Recall@10 by only 0.13 points, below the required 1 point; it saves 174 B but has about 28% lower load+scan throughput. Hard tying is only 0.03 points lower in Recall@10 and 109 B smaller. The static residual quantizer is smaller and more accurate than every learned variant in these worlds.

### Fresh audit results (mean ± sample SD across worlds)

| Method | nMSE | Recall@1 | Recall@10 | Model + packed codes | Payload saving vs raw DB | Load+scan QPS | Decode resident RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static residual quantizer | .06726 ± .00190 | 17.97 ± 2.07% | 53.07 ± .76% | 9,921 B | 84.9% | 182,264 ± 56,361 | 65,536 B |
| Independent QINCo | .08405 ± .00106 | 10.94 ± .78% | 47.81 ± 1.07% | 60,386 B | 7.9% | 125,228 ± 12,912 | 65,536 B |
| Hard-tied QINCoStep | .12602 ± .00788 | 5.99 ± 1.80% | 42.32 ± .61% | 22,525 B | 65.6% | 124,295 ± 3,123 | 65,536 B |
| Givens Mirror stage view | .12599 ± .00780 | 5.47 ± 1.56% | 42.34 ± .79% | 22,634 B | 65.4% | 93,725 ± 12,838 | 65,536 B |
| Additive rank-1 stage control | .12601 ± .00792 | 5.99 ± 1.80% | 42.21 ± .97% | 22,808 B | 65.2% | 130,620 ± 2,118 | 65,536 B |

All methods store fewer bytes than the raw float32 database. Every arm uses the same 2-byte/vector packed index code stream (plus nearly identical manifests); model state dominates learned methods. Static RQ wins here because the generated clusters are simple enough for explicit stage codebooks. This is a task-specific screen, not evidence against QINCo on real high-dimensional embeddings.

Each learned neural method consumes 5,120 train examples and 80 updates per world. Shared stagewise k-means initialization takes 0.0660, 0.0665 and 0.0680 s in the three audit worlds (five-run medians measured in a separate replay after metrics were frozen; initialization is common to every arm and excluded from per-arm train-wall figures). The encoder generator proxy is 196,608 MACs/vector for neural QINCo-style methods; a 512-row database therefore has about 100.7M candidate-generator MACs during encoding. Mirror's stage rotations add 288 scalar FLOPs/vector (147,456 per database encoding); additive conditioning adds 96 scalar additions/vector. One 128-query scan over 512 decoded rows uses about 2.10M vector-distance MACs. These counts exclude optimizer backward work and BLAS overhead; measured wall is also reported in the per-world JSON files.

## C — Strongest counter-hypothesis

The vector worlds are small and synthetic, K=16/M=4, with only 80 updates and k-means residual initialization. They are not the real high-dimensional retrieval distributions or training regimen for which QINCo was developed. The independent QINCo model is near the raw database size and underperforms static RQ here, showing that this workload is not a demanding test of implicit codebooks. Therefore this result establishes only that the proposed shared-stage Mirror does not pass this bounded screen; it does not establish a general QINCo limitation.

## U — Unconfirmed

- Real SIFT/Deep1B/BigANN or embedding retrieval and large-scale search/index integration.
- Official pretrained QINCo checkpoints and their intended initialization/training schedules.
- Larger dimensions, codebooks, stage counts, and trained random seeds beyond three fresh generated worlds.
- Whether a private-stage residual can recover native recall without erasing the saved model bytes.

## Fact / Interpretation / Hypothesis

- **Fact:** Across three fresh synthetic worlds, Mirror averages 42.34% Recall@10, 22,634 B serialized model+index, and 93,725 load+scan QPS; exact payload and metric replay pass.
- **Interpretation:** The stage coordinate saves model bytes against native independent QINCo but fails its quality/runtime gate, and it offers no meaningful quality edge over hard tying or additive conditioning. In this workload, ordinary explicit residual codebooks dominate.
- **Hypothesis:** Real embedding distributions may need independent stage functions or private residuals; if revisited, test those on a real retrieval dataset and charge runtime reconstruction.

## H / T / D / C / U

**H:** One shared QINCo residual generator plus per-stage Givens coordinates will yield useful logical stages with at least 25% lower payload, native-quality Recall@10, and a Mirror-specific edge over additive conditioning.

**T:** Pinned QINCo v1 model structure; five methods; K=16/M=4/32-D vectors; development world plus three fresh synthetic worlds; 80 updates/arm/world; exact top-10 ground truth; actual model+packed-index bytes; warmed repeated timing; metric replay.

**D:** FAIL. Storage passes versus independent QINCo, but quality, throughput and Mirror-versus-additive gates fail.

**C:** Synthetic cluster data and reduced CPU training may not exercise the setting where QINCo's conditional codebooks provide their published benefit.

**U:** Real-vector retrieval, official checkpoints/training schedules, larger models and private-stage residual frontiers.
