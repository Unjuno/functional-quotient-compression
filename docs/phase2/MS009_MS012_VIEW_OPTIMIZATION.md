# MS009–MS012 — Residual View decomposition, controls, combination, and global optimization

Date: 2026-10-07 JST  
Evidence lane: SYNTHETIC (small 4→48→48→2 MLP sensor worlds).  
Not a Transformer, real-sensor, natural-language, or general capacity result.

## Architecture contract

The shared world core receives only calibrated canonical state. Sensor identity is not provided as an ordinary core input. Sensor-specific computation is restricted to a small residual View.

Primary residual View:
- structured stretch or shear+stretch conjugation around GELU;
- deterministic sensor address;
- learned low-dimensional sensor code;
- no ES;
- no loop;
- Backprop for code and shared core.

## MS009 — matched low-description conditional controls

Nonlinear sensor mismatch beta=0.6. Fresh worlds 53701–53710.

After discovering a training-budget confound, the final panel used:
- shared parent: 1000 AdamW updates;
- learned View code-only stage: 400 updates;
- final joint core+View stage: 1000 updates;
- shared and fixed-gain controls: 1000 shared-core updates from the same parent;
- source0-only changed-law adaptation: 300 core-only updates;
- rank4, two hidden layers, 32 learned sensor-code scalars for every learned conditional control;
- actual uncompressed FP32 serialized bytes measured.

Median ratios versus shared-only:

| Method | Seen NMSE | Seen wins | Cross-sensor transfer NMSE | Transfer wins | Joint geometric ratio | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| Mirror combined | 0.8127 | 10/10 | 0.9767 | 7/10 | 0.8996 | 11,961 |
| Mirror stretch | 0.8525 | 10/10 | 0.9787 | 8/10 | 0.9108 | 11,960 |
| Learned direct gain | 0.9071 | 10/10 | 0.9930 | 8/10 | 0.9422 | 11,958 |
| Low-rank residual | 0.9562 | 10/10 | 0.9858 | 9/10 | 0.9691 | 11,953 |
| Fixed gain | 0.9524 | 10/10 | 1.0318 | 3/10 | 0.9883 | 11,696 |
| FiLM | 0.9745 | 10/10 | 1.0196 | 2/10 | 0.9954 | 11,950 |

Shared-only payload: 11,692 B.

Interpretation: conditional capacity in general helps, but the structured Mirror controls gave a larger seen-law improvement than same-code-count gain/FiLM/low-rank controls. Fixed gain improved local prediction but usually harmed cross-sensor law transfer.

## MS011 — combination optimization

New dev world 53900; unseen fresh worlds 53901–53910.

Question: with 64 learned View-code scalars, is it better to combine heterogeneous residual mechanisms or increase the breadth of one structured Mirror family?

Results versus shared-only:

| Method | Seen ratio | Seen wins | Transfer ratio | Transfer wins | Both improve | Joint ratio | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| stretch-r8 | 0.7317 | 10/10 | 0.9662 | 8/10 | 8/10 | 0.8338 | 12,088 |
| combined-r8 | 0.6298 | 10/10 | 1.0052 | 5/10 | 5/10 | 0.7966 | 12,089 |
| combined+low-rank | 0.7618 | 10/10 | 0.9981 | 5/10 | 5/10 | 0.8764 | 12,212 |
| gain-r8 | 0.8209 | 10/10 | 1.0330 | 2/10 | 2/10 | 0.9218 | 12,086 |
| low-rank-r8 | 0.8379 | 10/10 | 1.0123 | 3/10 | 3/10 | 0.9164 | 12,081 |
| gain+low-rank | 0.8151 | 10/10 | 1.0164 | 1/10 | 1/10 | 0.9100 | 12,209 |
| stretch-r4 | 0.8720 | 10/10 | 0.9860 | 7/10 | 7/10 | 0.9280 | 11,958 |
| combined-r4 | 0.8596 | 10/10 | 0.9967 | 5/10 | 5/10 | 0.9302 | 11,959 |

Near-byte direct comparisons:
- stretch-r8 vs gain-r8: seen median ratio 0.8789, 9/10 wins; transfer median ratio 0.9201, 10/10 wins; both metrics 9/10.
- stretch-r8 vs low-rank-r8: seen median ratio 0.8292, 9/10 wins; transfer median ratio 0.9493, 10/10 wins; both metrics 9/10.

Decision: for the shared-world objective, stretch-r8 is the best balanced current View. combined-r8 gives the strongest local prediction but has weaker transfer consistency.

### Runtime diagnostic

Tiny CPU fixture: AMD EPYC 9V74 virtual CPU, PyTorch 2.10 CPU, one thread, FP32 eager, batch96 mixed addresses, clock unlocked.

Median forward time:
- shared: 0.0420 ms
- low-rank-r8: 0.1617 ms
- gain-r8: 0.1880 ms
- stretch-r8: 0.2907 ms
- combined-r8: 0.3935 ms

Median forward+backward+SGD step:
- shared: 0.309 ms
- low-rank-r8: 0.672 ms
- gain-r8: 0.817 ms
- stretch-r8: 1.143 ms
- combined-r8: 1.496 ms

This is a tiny CPU implementation result, not a GPU prediction.

For a fixed sensor address, alpha=1 stretch-r8 can be folded into ordinary FFN weights. On one trained model:
- 16,384 / 16,384 argmax outputs matched;
- max logit absolute difference: 2.38e-7;
- native mixed-address forward: 0.292 ms median;
- grouped folded forward: 0.0978 ms median;
- 2.99x speedup on this CPU fixture;
- shared serialized payload: 12,088 B;
- four expanded expert weight sets: 43,040 B RAM.

Thus disk/storage sharing and deployment RAM/compute are separate resources.

## MS012 — global optimization of calibration + View + shared core

Architecture: stretch-r8. New dev world 54000; unseen worlds 54001–54005.

After decomposed View training and 1000 joint core+View updates, an additional 600-update global stage compared:
1. fixed calibration;
2. calibration freely optimized by task loss;
3. calibration optimized jointly with task loss plus synchronized paired-view alignment, lambda=0.1.

Median ratios versus fixed calibration:

| Global stage | Seen ratio | Seen wins | Transfer ratio | Transfer wins | New-sensor ratio | New-sensor wins | Both seen+transfer improve |
|---|---:|---:|---:|---:|---:|---:|---:|
| Free task-only calibration | 0.9222 | 5/5 | 1.0492 | 0/5 | 0.9926 | 5/5 | 0/5 |
| Paired-view alignment lambda=0.1 | 0.9836 | 5/5 | 0.9957 | 4/5 | 0.9965 | 5/5 | 4/5 |

Free calibration consistently improved local prediction while consistently degrading source-only law transfer. Paired-view alignment sharply reduced calibration drift and preserved most transfer.

## Current design rule

The evidence supports this hierarchy:

1. calibrate/encode each sensor toward a canonical state;
2. do not expose raw sensor identity to the shared world core;
3. use no View when calibration residual is small;
4. when irreducible observation mismatch remains, use a low-description residual View;
5. prefer a single sufficiently broad structured Mirror family over summing heterogeneous residual mechanisms;
6. stretch-r8 is currently the best balanced candidate;
7. Backprop trains View codes and shared weights;
8. keep a synchronized alignment objective if calibration is allowed to move during end-to-end training;
9. count all View/calibration metadata in serialized bytes;
10. keep local prediction, cross-sensor law transfer, new-sensor onboarding, compute, and storage as separate metrics.

## Mirror-MoE status

The current validated residual-View architecture should **not** be mistaken for sparse MoE merely because multiple sensors/views can be evaluated in parallel.

Mirror-MoE remains an active, separate hypothesis:

- keep one shared world core;
- maintain a bank of low-description residual Mirror experts generated from shared parameters;
- route from canonical world state / role, not raw sensor identity;
- activate only top-1/top-2 experts so active compute does not scale linearly with the total expert count;
- compare against fixed gate, FiLM/low-rank residual, dense parallel Mirror bank, and independent MoE under actual serialized-byte and active-compute budgets.

See [MIRROR_MOE_OPEN_DIRECTION.md](MIRROR_MOE_OPEN_DIRECTION.md).

The key unresolved question is whether sparse Mirror selection preserves the shared-law-transfer advantage while obtaining specialization at approximately constant active compute.

## Boundaries

These are small synthetic results. Fresh-world ranges are not confidence intervals. The experiments do not establish that world laws are stored only once internally, do not demonstrate a Transformer capacity multiplier, and do not establish real camera/LiDAR/IMU performance.
