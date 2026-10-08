# SFM003 — One heavy forward, four/five learned useful outputs

Date: 2026-10-08 JST. Research-only add-on to **MA-1175**. Outcome: **aligned synthetic mechanism PASS; Mirror-specific novelty M0; independent off-orbit specialist reconstruction FAIL**. This is *not* a newly completed MA scientific status, not a pretrained LLM experiment, and not a GPU offloading study.

## 1. Central thesis and causal mechanism

A single expensive shared statistic `z(x) = GELU(x @ U) @ D` is evaluated **once** for every input; K cheap learned Mirror code transformations produce distinct predicted functions:

`hat{y}_j = T_{m_j}(z(x))`, `j=1,...,K`.

The teacher generates `y_j(x) = T_{a_j}(z(x)) + beta * g_j(x)` where `T_a` rotates four independent 2-D output planes and `g_j(x)` is an independently drawn nonlinear private function, normalized with statistics from task-training examples only. The design explicitly supports a favorable orbit-aligned condition `beta=0`, and two off-orbit stress conditions `beta=.3` and `.8`.

The true teacher `a_j` and `g_j` weights are **not given to training procedures**. The shared trunk is frozen and exactly shared with the data-generating teacher: this isolates whether lightweight task heads can *learn* role transformations and does not prove the joint-training problem or natural-task sharing.

## 2. Preregistration and environment

[Frozen protocol](pilots/sfm003_trained_multioutput/PROTOCOL.json) was committed at `e6f6bf6708a63480c17a11e9a8b43408df53e494` before development and fresh evaluation. CPU PyTorch 2.10.0+cpu, NumPy 2.3.5; one PyTorch thread, FP32, no CUDA. Dev seeds `11,12,13` and fresh seeds `101,102,103,104,105`. Input 16, shared hidden 64, output 8. K in `{4,5}`, train examples 768, dev 256, fresh 512. Adam 350 whole-batch updates, lr .04, no weight decay; source-trunk weights are fixed. For timing batch 128, 20 warmups, 100 timed calls per world/method.

**Data firewall:** output-head training sees only 768 training inputs and corresponding task targets. A separate independent fresh world is generated for each seed. Audit examples/targets do not drive optimization, ranking, or early stopping. No hyperparameters changed between the development and fresh runs. No downstream LM NLL, natural task, GPU model or PCIe measurements.

## 3. Measured held-out quality (median over five fresh worlds)

| K | Private beta | Mirror 4-scalar/head NMSE | Diagonal+Bias NMSE | Full linear head+Bias NMSE | Shared same-output NMSE |
|---:|---:|---:|---:|---:|---:|
| 4 | 0 | <= 1.12e-15 | 0.18065 | 0.00000 | 0.28387 |
| 5 | 0 | <= 7.09e-16 | 0.17581 | 0.00000 | 0.23588 |
| 4 | .3 | 0.09622 | 0.25467 | 0.06993 | 0.37215 |
| 5 | .3 | 0.09338 | 0.25606 | 0.07151 | 0.31012 |
| 4 | .8 | 0.42179 | 0.52107 | 0.30708 | 0.59865 |
| 5 | .8 | 0.41833 | 0.51722 | 0.32182 | 0.57709 |

NMSE is `mean squared error / centered held-out target variance` **per task**, averaged over K tasks. Smaller is better. Full linear head is a strong non-Mirror output-code baseline that also executes the shared trunk **only once**, and typically fits the off-orbit targets better than Mirror. At beta=0 both linear and Mirror capture all tasks on this favorable constructed family. Ordinary pairwise-rotation output heads are **mathematically identical** to the Mirror used here, and are therefore a native M0 null on function/bytes/compute.

Distinctness test: at beta=0 the ratio of the minimum pairwise distance between predicted heads to the corresponding target distance is 1.0 across all five seeds for both K=4 and 5. The no-View same-output control has zero pairwise prediction separation, demonstrating that trivial collapse is rejected. At beta=.8, the Mirror pairwise distance ratio falls substantially (K4 median .394, K5 median .209), exposing a genuine functional limit rather than merely worse parameter-fitting accuracy.

## 4. Physical bytes and CPU latency (K=4/5, beta=0)

| Method | K4 actual named NPZ bytes | K5 actual named NPZ bytes | K4 CPU median time | K5 CPU median time |
|---|---:|---:|---:|---:|
| Shared with no per-head change | 6,634 B | 6,634 B | 0.02781 ms | 0.02790 ms |
| **Learned output Mirror** | **6,942 B** | **6,958 B** | **0.06477 ms** | **0.06748 ms** |
| Ordinary diagonal+bias | 7,372 B | 7,436 B | 0.03569 ms | 0.03630 ms |
| Ordinary full linear head+bias | 8,268 B | 8,556 B | 0.04848 ms | 0.04854 ms |
| Naively compute the heavy shared trunk K separate times + rotate | same learned code (but redundant compute) | same learned code | 0.20836 ms | 0.26046 ms |

The last row intentionally models **unnecessary** repeated heavy work; it is not the strongest native architecture. The actual per-world median time ratio of redundant K heavy forwards over shared Mirror was **3.20x at K4** and **3.87x at K5**, with ratio >=1.5 in 5/5 worlds at both K values. The learned rotation readout was approximately 1.34x to 1.39x **slower** than a regular shared-trunk linear-head method in this CPU eager implementation. Thus there is no standalone Mirror-specific runtime win over proper shared-trunk native multi-head execution.

The whole-model named NPZ storage of the K5 Mirror system is 6,958 B, versus 8,556 B for ordinary separate linear readouts, **18.68% smaller** in this 16->64->8 toy; that advantage trades away native linear-head off-orbit expressivity. Both models share the same dominant heavy U,D tensors. Raw angle code is 4 FP32 values/head; serialized bytes include NPZ array headers. GPU fused kernel efficiency and real tokenizer/embedding bytes are not covered.

## 5. Decision vs frozen gates

- **Aligned orbit mechanism:** PASS: beta0 mean audit NMSE <=1.12e-15 for K4 and <=7.09e-16 for K5, both well under .01 threshold; all 5/5 worlds per K preserve noncollapsed output distances. One heavy forward, K cheap learned heads.
- **CPU improvement vs intentionally redundant K-forward reference:** PASS (5/5 worlds; 3.20x / 3.87x median, thresholds >=1.5x).
- **Out-of-orbit private skills:** FAIL as desired falsification: beta .8 median Mirror NMSE .422 (K4) and .418 (K5), well above .1, while full linear heads are better but not perfect.
- **Mirror-specific speed or novel function class over native paired-Givens:** M0; exact model equivalence. Native shared linear head runs faster and also one-forwards the trunk.
- **Real-model adoption:** UNCERTAIN/NOT TESTED. No trained shared trunk with multiple independent functions, no real LLM/embedding task, no GPU transfer overlap. Do not promote MA-1175 beyond UNTESTED based only on this pilot.

## 6. Primary next experiment: learned shared trunk (the real question)

Train ONE shared trunk jointly from actual K distinct task targets, without giving teacher canonical `z`, and evaluate different task correlations/interference at the same physical inference bytes. Compare: (A) shared trunk+ordinary linear heads, (B) shared trunk+Mirror output code, (C) shared trunk+nonlinear shared preactivation + per-View light tails, (D) native BatchEnsemble/MIMO and matched parameter superposition, (E) independent K teachers. Measure quality per K and amortized active FLOPs, P50/P95, memory/IO and counterfactual ablations. Reject trivially learned rotations if per-view useful accuracy or independent task competence fails.

For GPU offload, MA-1176 independently examines pipelining the next physical group while current 4/5 head outputs are generated; compute hiding depends on actual GPU occupancy, PCIe bandwidth and router prediction, and is not inferred from these CPU numbers.

## H/T/D/C/U interpretation and units

H: shared heavy execution can support several related useful functions at low extra code cost; the benefit is conditional on shared representation sufficiency. T: the frozen K×beta×seed method matrix above. D: PASS/M0/FAIL as scoped. C: independent expert functions outside the output View's representable family require a richer head/private update; ordinary multi-output architecture already shares the trunk. U: five world means are not population-level intervals; timing is clock-unlocked CPU eager and method ordering may vary with hardware, kernel and batch.

| Variable | Meaning | SI unit / practical | Definition / type |
|---|---|---|---|
| x | input | 1 | real vector length16 |
| z | shared heavy result | 1 | real vector length8 |
| m_j | learned task code | 1 (angle rad) | 4-vector each for pairwise Givens |
| y_j | desired task output | 1 | real vector length8 |
| K | number of outputs | 1 | integer 4/5 |
| beta | off-orbit private strength | 1 | nonnegative real 0/.3/.8 |
| L | mean NMSE | 1 | dimensionless nonnegative scalar |
| S | serialized storage | byte (non-SI practical) | actual named NPZ bytes integer |
| t | observed CPU inference time | s | positive real, ms in tables |

`z @ W_head` conforms only when W_head has an 8-dimensional input axis; 4 rotation angles act on 4 disjoint 2-D output pairs. Relative NMSE is squared error divided by squared target scale, dimensionless. Latency and byte storage have distinct units and cannot be added.

## Provenance

See [source](pilots/sfm003_trained_multioutput/source/run_sfm003.py), [unit tests](pilots/sfm003_trained_multioutput/source/test_sfm003.py), [dev quality](pilots/sfm003_trained_multioutput/results/dev_quality.csv), [fresh quality](pilots/sfm003_trained_multioutput/results/fresh_quality.csv), [dev timing](pilots/sfm003_trained_multioutput/results/dev_timing.csv), [fresh timing](pilots/sfm003_trained_multioutput/results/fresh_timing.csv), and [hashes/environment](pilots/sfm003_trained_multioutput/results/VERIFICATION.json). The scientific status of the official worker and main remains unchanged.
