# MLP tail exact rate/support boundary (T215–T220)

This note records durable conclusions from local experiments T215–T220. The functional proxy in T217 is synthetic and is intentionally not promoted as real-model quality evidence.

## Exact 64x candidate bytes

Under the current 28M whole-model rate skeleton with a 32 KiB metadata reserve and 1,624,624-byte 64x target, exact packed tail sections and full placeholder containers were built and round-tripped:

| tail K | protected neuron bundles | coverage | whole bytes | headroom |
|---:|---:|---:|---:|---:|
| 256 | 1,976 | 12.0605% | 1,624,599 | 25 |
| 512 | 1,542 | 9.4116% | 1,624,624 | 0 |
| 1024 | 959 | 5.8533% | 1,624,600 | 24 |

The full placeholder containers include the previously reserved 32 KiB metadata region. A concrete 128-entry directory with per-section CRC32 and payload SHA-256 occupies 4,352 bytes of that reserve; 28,416 bytes remain reserved. K512 therefore fits exactly at the hard target but has no byte headroom outside the reserve.

These are exact byte-layout / serializer results over deterministic placeholder payload, not real encoded weights.

## Self-trained tail codebook support boundary

The current coupled-neuron tail design protects one MLP neuron bundle consisting of one FC output row plus the corresponding PROJ input column. At hidden size 512 this is 1,024 scalars = 32 residual blocks of 32 scalars.

For a self-trained selected-tail K-codebook, a necessary condition is therefore

`protected_bundles * 32 >= K`

in addition to exact rate fit.

Relevant results:

- K1024: 959 bundles -> 30,688 residual blocks -> support ratio 29.97 -> feasible.
- K2048: the byte budget leaves only 59 bundles -> 1,888 residual blocks < 2,048 -> infeasible under the current self-trained selected-tail design.
- K4096: codebook metadata alone exceeds the available tail budget.

Therefore **K1024 is the largest feasible power-of-two K in the current self-trained tail format**. K2048/K4096 should not be explored further unless the codec design changes, e.g. by training from external/shared residual samples; that would be a different representation contract.

## Real-model run gate

The three remaining candidates are K256, K512, and K1024. All have exact RATE_PASS and the required training support, but all remain **REAL_QUALITY_PENDING**.

When the real TinyStories-28M checkpoint is available, the run order is:

1. verify checkpoint provenance and MLP tensor shapes;
2. build real K64 shared MLP base and real uniform-K128 baseline;
3. freeze first-order selected bundle sets for K1024/K512/K256;
4. run an exact-restoration upper bound for each selected set;
5. prune a candidate immediately if exact restoration cannot beat uniform K128 on holdout;
6. train/quantize residual tail only for survivors;
7. serialize emitted bytes;
8. require whole-model JOINT replay before any real quality pass claim.

Historical 8M layer2 c_fc K64/K128/K256 KL values are only a boundary reference and must not be treated as 28M candidate scores.

## Evidence boundary

No TinyStories checkpoint weights were available during T215–T220. T215/T216/T218/T219/T220 are exact serializer, necessary-feasibility, or software evidence. T217 used a nonlinear synthetic proxy only. No new real-model functional quality claim is made here.
