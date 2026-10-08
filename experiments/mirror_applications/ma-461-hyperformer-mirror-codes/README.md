# MA-461 — HyperFormer generator outputs Mirror codes

Status: **FAIL** (development screen; fresh seeds sealed)  
Evidence lane: MECHANISM/STORAGE/GENERALIZATION/RUNTIME  
Branch: `research/ma-461-hyperformer-mirror-codes-20261008`  
Base commit: `a211269`  
Prior art: PA82 (HyperFormer)

## H — Hypothesis

A HyperFormer-style generator emitting a two-value code, decoded through a shared matrix basis, could match a generator emitting all 16 adapter values while reducing actual bytes and generation compute on heldout task×layer combinations. A direct native low-rank generator tests whether the gain is Mirror-specific.

## T — What ran

The frozen mechanism screen has six task identities × four layer positions. Eighteen combinations train, and six are held out; each task and layer appears in training. The teacher is a smooth rank-two adapter family on a shared identity base. Both generated models use the same learned 2D task and layer embedding tables and a 16-wide tanh hidden layer. HyperFormer emits 16 adapter values; Mirror emits two codes and decodes them through two shared 4×4 matrices. Both ran 1,800 Adam updates with eight sampled combinations per update. Each heldout pair uses 256 input vectors. Development seeds: 46101 and 46102. A separate full table is trained only on seen pairs; it is not a valid heldout predictor and is reported as a seen-pair storage/fit reference. Fresh seeds 46111–46113 remained sealed.

Every payload includes the base matrix, learned task/layer embeddings, generator weights and biases, Mirror basis where used, and shape/format metadata. `RESULTS_CORE.csv` records actual uncompressed `.npz` bytes and all run metrics.

## Results — facts

| Seed | Method | Heldout output RMSE | Seen-pair RMSE | Actual bytes | MAC proxy/input | Train wall (s) |
|---:|---|---:|---:|---:|---:|---:|
| 46101 | HyperFormer full output | 0.02641 | 0.01311 | 3,854 | 352 | 4.423 |
| 46101 | Mirror 2-code generator | 0.02944 | 0.02149 | 3,286 | 160 | 6.570 |
| 46101 | native low-rank generator | 0.02944 | 0.02149 | 3,286 | 160 | 6.785 |
| 46101 | independent seen-pair table | — | 0.00113 | 2,852 | 16 | 2.255 |
| 46101 | shared adapter | 0.23183 | 0.24597 | 1,080 | 16 | 2.151 |
| 46102 | HyperFormer full output | 0.04120 | 0.00801 | 3,854 | 352 | 5.905 |
| 46102 | Mirror 2-code generator | 0.07078 | 0.00461 | 3,286 | 160 | 8.048 |
| 46102 | native low-rank generator | 0.07078 | 0.00461 | 3,286 | 160 | 8.612 |
| 46102 | independent seen-pair table | — | 0.00062 | 2,852 | 16 | 2.542 |
| 46102 | shared adapter | 0.18768 | 0.19369 | 1,080 | 16 | 1.798 |

Mirror used 14.7% fewer actual bytes than HyperFormer (3,286 vs 3,854), short of the frozen 25% target. Its MAC proxy was 160 vs 352, a 54.5% reduction. Heldout RMSE was worse than HyperFormer in both seeds; seed 46102 Mirror exceeded the 0.05 quality bound. Mirror's train wall was also slower in both runs despite its smaller proxy. The query timing covers only six small CPU batches and is not stable deployment evidence.

The direct native low-rank generator had exactly the same heldout outputs, serialized bytes and hash as Mirror in each seed. The seen-pair full table fit its observed pairs better and was smaller in this small setup, but it cannot predict the six heldout combinations; it is not a fair generalization winner. The shared adapter performed poorly. The initial delta-only output metrics and their payloads remain preserved under `runs/superseded_delta_only_metric_*`; corrected runs explicitly apply the paid identity base plus adapter and reproduce the same error values.

## D — Decision

**FAIL.** The frozen gate is missed on storage and heldout quality, and Mirror is exactly a native low-rank hypernetwork parameterization. It shows a compute-proxy reduction and a modest byte reduction against the full-output generator, but the heldout-quality loss, slower measured training, and exact native alias prevent a Mirror-specific or Pareto claim. Fresh combinations were not opened.

## C — Strongest counter-hypothesis

The only useful effect is the ordinary rank-two shared-basis hypernetwork. The identical native control reproduces every Mirror prediction and byte. The teacher is itself rank two, so it favors compressed coordinates; even there, the full HyperFormer generalizes better on both seeds.

## U — Unconfirmed

This is an adapter-matrix mechanism task, not a trained Transformer or language evaluation. It does not establish near-convergence capacity, task-level transfer across new task identities, natural heldout data, or stable serving latency. Fresh seeds were sealed by the frozen gate.
