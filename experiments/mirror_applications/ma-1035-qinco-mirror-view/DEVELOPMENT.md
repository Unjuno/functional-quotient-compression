# MA-1035 development record

## Fact

Development world seed 103500 used 1,024 train vectors, 256 validation vectors, 512 database vectors and 128 queries in 32 dimensions. Each world has 16 newly sampled Gaussian cluster centers with anisotropic noise. Model settings were K=16, M=4, one residual block, hidden size 32, 80 Adam updates, batch 64, lr 0.001. Audit-world generation seeds remain unopened.

| Method | nMSE | Recall@1 | Recall@10 | Model + codes | Train wall (s) | Encode (s) | Decode/load (s) | Scan QPS, median | Load+scan QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static residual quantizer | 0.07820 | 0.21094 | 0.53750 | 9,921 B | 0 | 0.00074 | 0.00014 | 207,320 | 169,840 |
| Independent QINCo | 0.09462 | 0.16406 | 0.50391 | 60,386 B | 0.336 | 0.03287 | 0.00042 | 210,361 | 124,588 |
| Hard-tied QINCoStep | 0.13534 | 0.09375 | 0.41797 | 22,525 B | 0.264 | 0.01059 | 0.00042 | 224,716 | 128,748 |
| Givens Mirror stage view | 0.13520 | 0.09375 | 0.42266 | 22,634 B | 0.344 | 0.01451 | 0.00078 | 227,536 | 95,379 |
| Additive rank-1 stage control | 0.13529 | 0.09375 | 0.42266 | 22,808 B | 0.348 | 0.01289 | 0.00045 | 234,316 | 129,106 |

All learned methods used 80 updates and 5,120 train examples. Database code stream is two packed bytes/vector plus its manifest. Database decode timing is the median of 20 runs; exact 128-query L2 scan timing is the median of 20 scans after warm-up; encode timing is the median of five complete encodes. The raw single-shot timing run is preserved in `runs/development_unwarmed_103500.json`; no quality or storage setting changed.

## Interpretation

Mirror is 0.47 percentage points above hard tying in Recall@10, exactly equal to additive control on this world, and 8.12 points below independent QINCo. Its total serialized bytes are 62.5% below independent QINCo but only 109 B above hard tying; it is 174 B smaller than additive at equal measured Recall@10. The static RQ control is both much smaller and more accurate than every learned generator on this world. Median load+scan throughput for Mirror is 26% below additive in this one world; the three fresh worlds will show whether that difference persists.

## Hypothesis

The shared stage-generator family will likely fail native-QINCo quality tolerance and may be dominated by static residual codebooks on these simple synthetic clusters. The three fresh worlds determine whether this is a stable scoped failure.
