# MS003 / SN001: mechanism audit and synchronized sensor bridge

Date: 2026-10-07 JST. Evidence: mathematical identity, recovered-model interventions, small synthetic sensor pilot. NOT a Transformer, real sensor, capacity, or equal-compute result.

## Established mechanism

A fixed-address Mirror is an ordinary expert with constrained, tied weights. For a two-hidden-layer MLP, the effective weights are Q1 W1, Q2 W2 inverse(Q1), and W3 inverse(Q2); hidden biases are Q1 b1 and Q2 b2. Here W are shared real weight matrices, b are biases, and Q are fixed invertible matrices. All features and coefficients are dimensionless. These effective experts cannot be independently chosen.

Componentwise GELU commutes with a pure permutation. Consequently permutation -> GELU -> inverse permutation is exactly GELU. Earlier suggestions that this placement of pure permutations would preserve useful specialization are withdrawn. Signed transforms are different; GELU is not odd.

A 2D shear mixes the second feature into the first before GELU and subtracts its separately activated contribution afterwards. It changes both the nonlinear response and the Jacobian, not just stored parameter count. A fixed view can be folded for inference; training must still respect the shared parameterization.

## Recovered SM002 audit

The original SM002 ZIP passed CRC and all 1,949 manifest hashes. Six Mirror checkpoints (private16 and private32, worlds22001/22002/22003 and seeds201/202/203) were expanded into four ordinary experts. All 21,504 finite-domain predictions matched. Maximum logit differences: FP32 6.866455078125e-05, FP64 1.4210854715202004e-13. Maximum relative L2 error between gradients through folding and original shared parameters: 4.877905772537196e-16.

Native serialized model: 44,155 B. Expanded four-expert model: 175,114 B. This expansion is built from already-tied weights; it is NOT evidence that arbitrary independent experts compress fourfold.

At private32, median private accuracy was 72.63%; removing both shears without retraining reduced it to 6.08%, and a cyclic wrong view to 4.03%. Reliance on a learned operator is not by itself superiority over a control.

Fixed-view inference microbenchmark: AMD EPYC 9V74 CPU, one thread, FP32, PyTorch2.10.0+cpu, precomputed batch128 of 72D features, 30 warmups, 200 forwards per group, 7 groups. Native Mirror median 0.247126 ms [0.242950,0.266006]; folded cached expert 0.084513 ms [0.084313,0.087271]. Clock not locked; snapshot approximately2596 MHz. Expansion cost excluded. This is NOT training speed or mixed-view throughput. Caching expanded experts costs RAM.

## SN001 protocol

Four full-rank affine sensors observe the same normalized four-dimensional position/velocity state with Gaussian noise sd0.01. Target: two-dimensional synthetic nonlinear acceleration. Sensor matrices are random QR/scaled transforms, not generated to match Mirror shears. The reference sensor0 has identity observation. This does not cover partial observability, images, lidar, IMU hardware, temporal attention or missing sensors.

Train sensors0/1/2 only. Main1024 latent events; separate256 synchronized calibration events;4096 independent test events; separate1024 adaptation events. Calibration uses paired sensor readings to sensor0, not acceleration labels. New sensor3 never supplies task labels.

Eight controls: raw/shared, raw/Mirror, raw/fixed-gate, calibrated/shared, calibrated/Mirror, calibrated/gate, shuffled-pair calibration/shared, and true-inverse oracle/shared. All have two width48 hidden layers,4 observations plus4 sensor-ID inputs,2 outputs, tanh-GELU. Main1500 AdamW updates, batch96, cosine lr0.003 to0.0003, wd0.0001, clip1. Same core initialization and event stream within a world. Worlds53001/53002/53003, initial seeds401/402/403. No tuning or stopping based on test scores. Protocol SHA256:2368625bd0d2e81b0b7b7c0f959fc0551acc5701d37e4543425fcb9a6ed3ef98.

Source-only adaptation changes the common acceleration law but supplies extra observations and labels only through sensor0.300 updates, batch96, reset AdamW for every method, cosine lr0.001 to0.0001; calibration frozen. This measures transfer to sensors1/2, not simultaneous retention of old and new laws.

NMSE = summed squared prediction residual divided by summed squared centered true output, dimensionless, lower is better. Values below are medians of three paired sensor-world/initialization settings, not confidence intervals.

|Method|Seen-sensor NMSE|New sensor calibrated to reference address0|Receiver NMSE after source0-only law update|
|---|---:|---:|---:|
|raw shared|0.001562|0.001744|0.633255|
|raw Mirror|0.000608|0.000705|0.392593|
|raw fixed gate|0.001263|0.001570|0.649902|
|calibrated shared|0.000952|0.000947|0.011283|
|calibrated Mirror|0.000508|0.000582|0.037416|
|calibrated fixed gate|0.000898|0.000908|0.028741|
|shuffled-pair calibration|0.073421|1.124374|0.445056|
|oracle inverse|0.000953|0.000935|0.011375|

Calibrated Mirror and calibrated shared serialize to12,458 B, calibrated gate12,456 B, including all four paid calibration maps. Raw native models use12,068-12,070 B; their calibrated new-sensor deployment adds a151 B converter. This is a same-backbone/update comparison, NOT equal runtime/FLOPs.

Calibrated Mirror lowered seen-sensor NMSE by at least10% relative to calibrated shared and gate in all three settings. However calibrated shared transferred the changed law better: receiver errors fell95.92-98.03%, versus88.54-92.89% for calibrated Mirror. Calibrated gate fell93.56-95.72%. No raw or shuffled method achieved50% receiver-error reduction in all three settings.

A new sensor is not accepted merely by inventing a new internal view. Post-hoc address interventions on calibrated Mirror: median new-sensor NMSE0.000582 with input ID0/hidden view0;0.070950 with input ID0/view3;0.104216 with input ID3/view0. Both sensor-ID features and internal-view unfamiliarity matter. These interventions selected no candidates and performed no retraining.

## Decision and boundaries

PASS for the preregistered narrow sensor-readiness and fixed-update Mirror-accuracy gates. Transfer PASS for calibrated variants and oracle, FAIL for raw/shuffled controls. Capacity and equal-compute benefit NOT TESTED. Calibration plus shared dynamics is the next sensor baseline; stronger specialization is not automatically stronger physical-law transfer.

Executed24 main trainings,24 source-only adaptations,6 exact full main-training replays.13 unit tests;120 exported sensor checkpoints reloaded with exact predictions and metrics; train/cal/test/adapt event intersections empty. This is current-session self-audit, not external replication. Full source, original selected checkpoints, new checkpoints, data, protocol, Japanese derivations and audit logs are preserved in the conversation artifact MS003_SN001_RESULTS_2026-10-07.zip. This repository entry preserves core results and the exact folding implementation; historical MN/MT/RF/SM documents and main are untouched.

Related primary literature: GELU arXiv:1606.08415; sparse MoE arXiv:1701.06538; HyperNetworks arXiv:1609.09106; FiLM arXiv:1709.07871; Contrastive Multiview Coding arXiv:1906.05849. No claim to reproduce those benchmarks or to establish architectural novelty.
