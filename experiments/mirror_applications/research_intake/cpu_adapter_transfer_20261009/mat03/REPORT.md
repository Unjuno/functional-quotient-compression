# MAT03 / MAT03R — genuine same-input, one-heavy-forward, five-useful-output evidence

**2026-10-09 CPU experiment report. Scope: sklearn handwritten digits, five binary tasks on exactly the same original input image.** This repairs the input-transformation confound in MAT01/MAT02. It is NOT a natural language model, native MIMO/TabM/LoGo paper replication, GPU study, independent biological/linguistic function capacity theorem or Mirror-specific adoption result.

## Task, method, and evidence firewall

- Every original 8×8 image is processed **once** by the same trainable shared trunk 64→128 GELU→32 GELU; a forward hook verified **exactly one trunk forward** per batch across all nine methods.
- Each input has **five supervised binary outputs**: digit parity, digit >=5, horizontal pixel-mass imbalance, vertical mass imbalance, center pixel density. The latter three use thresholds estimated **only from the training images**, not audit images. All methods see identical normalized input pixels and all five labels. The five tasks are useful for this synthetic/observed classification fixture but do not prove independent generative skills or language capabilities.
- **Native baselines:** ordinary five linear output heads on one trunk, shared output head with zero role conditioning, BatchEnsemble/IA3-like per-role 32D gate, four-coefficient fixed/original feature diagonal gate, ordinary fixed-Hadamard affine m basis, learned linear 4×32 dictionary m bank, and rank1 paid private corrections. The stronger native one-trunk five-head baseline is NOT charged five heavy forward passes.
- **Mirror m:** a 4D task code controls \(Q_m=H\operatorname{diag}(\exp(Bm))H^T\),\ where H is a deterministic normalized Hadamard32 and B is a deterministic 32×4 source-independent mixing basis; the common final classifier is shared. The sole View-specific work is small-code scaling and 32D readout after the one heavy trunk call. **This is mathematically a structured feature modulation**, closely related to established IA3/FiLM; it must beat an ordinary same-byte affine role code before calling its geometry useful.
- **Fair training:** identical trunk initialization by world seed, 400 AdamW updates and 128-input batches/method, same data/update sampling per method, one-thread CPU. Five tasks trained end-to-end (not an oracle-provided common intermediate). Small rank1 private heads in two conditions are fully paid.
- **Serializer:** actual deterministic named-NPZ inference artifact, including trunk, task-specific readouts/codes, training-only preprocessing and task-threshold/config metadata; the fixed Hadamard/mixing generators use declared deterministic algorithm + seed. Byte differences on whole model are naturally modest because trunk dominates.

## Initial MAT03 [301..305] — exploratory / initial registered protocol

- Initial MA protocol preregistered at GitHub commit `52afafbae5b4f1888e632488f85c430e01d749fa` before task data audit.
- The actual MAT03 source was **not remotely committed before this initial fresh evaluation**. Do not describe its 301..305 evidence as fully code-frozen independent replication. The source was subsequently pinned for MAT03R.
- 3 development worlds [31,32,33], 5 initial fresh worlds [301..305]; nine complete methods per seed, all five task targets.

| method | mean NLL ↓ | accuracy ↑ | whole NPZ B ↓ | CPU p95 ms ↓ |
|---|---:|---:|---:|---:|
| native_multihead | 0.070840 | 0.9739 | 54220 | 0.2251 |
| mirror_private1 | 0.073240 | 0.9734 | 54651 | 0.4742 |
| native_linear_private1 | 0.073084 | 0.9734 | 54658 | 0.5246 |
| native_fastweight | 0.093286 | 0.9652 | 54337 | 0.1726 |
| mirror_hadamard4 | 0.125054 | 0.9550 | 53488 | 0.2454 |
| native_hadamard_linear4 | 0.149450 | 0.9417 | 53495 | 0.2501 |
| native_direct_diagonal4 | 0.222056 | 0.9026 | 53495 | 0.2361 |
| native_learned_dictionary4 | 0.305992 | 0.8658 | 54278 | 0.2666 |
| shared | 0.573914 | 0.7027 | 53156 | 0.1938 |

In the initial five fresh repeated partitions, Mirror4 vs same-code linear4 NLL differences (Mirror minus linear, negative favors Mirror): **-0.00703, -0.02821, -0.01748, -0.04438, -0.02489**; 5/5 negative. Mirror4 NLL ~0.125 vs linear4 ~0.149, but native full one-trunk five heads reaches ~0.071 and still executes one trunk call. Mirror-only saves only ~1.35% **whole inference NPZ bytes** (not 10%) relative to full five-head native and misses its task quality. Mirror+private rank1 reaches ~0.073 NLL, but requires **more total bytes** and more inference time than native five-head. This does not establish a Mirror-specific Pareto advantage.

## Independently preregistered post-discovery replication MAT03R [401..405]

- After seeing the above effect, pin the exact execution source SHA256 `e63b0f6cdc1f66d50cdd00629c05b4d0b800b73b9d24095a24076cad3736ffbe` at commit `825c281dbdcda67168a3c98289a7cc04a802f631`.
- Freeze new replication seeds [401..405], unchanged source, same 400 steps and all nine baseline methods, and an explicit **>=0.01 nat/example Mirror gain over linear4 in >=4/5 fresh worlds** at pre-audit commit `c6a1a16974a607b51fc121f4ca57f75fc634942b`. No new dev optimization on these seeds.

| method | mean NLL ↓ | accuracy ↑ | whole NPZ B ↓ | CPU p95 ms ↓ |
|---|---:|---:|---:|---:|
| native_multihead | 0.074660 | 0.9716 | 54221 | 0.1977 |
| mirror_private1 | 0.082487 | 0.9698 | 54652 | 0.5638 |
| native_linear_private1 | 0.081143 | 0.9687 | 54659 | 0.5304 |
| native_fastweight | 0.097402 | 0.9618 | 54338 | 0.1830 |
| mirror_hadamard4 | 0.156272 | 0.9471 | 53489 | 0.2611 |
| native_hadamard_linear4 | 0.157096 | 0.9391 | 53496 | 0.2233 |
| native_direct_diagonal4 | 0.242263 | 0.8973 | 53496 | 0.2219 |
| native_learned_dictionary4 | 0.273653 | 0.8915 | 54279 | 0.2436 |
| shared | 0.577559 | 0.6976 | 53157 | 0.1758 |

Paired Mirror4 minus affine linear4 NLL by fresh world **-0.00137, -0.00823, -0.00481, -0.01228, +0.02256**. Mirror has the lower error in 4/5 worlds but **only 1/5** meets the frozen 0.01 nat/label threshold. Mean improvement is just **0.000824** nat/example, with one reversal (+0.02256). Therefore **MAT03R H1 gate FAIL**.

- Native multihead mean NLL **0.074660** vs Mirror4 **0.156272**: no quality noninferiority. Whole NPZ around 53.5–54.2 kB gives only ~1.35% savings for Mirror4 versus native five heads. **H2 FAIL.**
- Mirror+private rank1 is closer in loss but needs more bytes and higher P95 than native five-head. On MAT03R it is slightly **worse than the ordinary same-code linear+private1 in 5/5 fresh worlds**, mean paired difference +0.001344 nat/label. **H3 FAIL.**
- Output/label diversity and single shared trunk are genuine and checked in 10 shape/data/gradient unit tests; **H4 PASS mechanical only**. The five tasks are binary classification on one fixed real-image dataset, not multiple independent LLM expert functions.

## Decision, counterexplanations and ERROR CHECK

**Strong conclusion:** Useful multi-output inference from one shared expensive trunk is possible and already realized by all the successful native methods too. A short structured 4D Mirror chart improves over some extremely small affine/diagonal controls in the first five splits, but independent frozen-code replication does **not** support the preregistered margin. Native ordinary multihead is both more accurate and much faster, at modest whole-model storage increase. No Mirror-specific Pareto superiority is demonstrated; the full MA-1175 scientific status remains UNTESTED.

- Different outputs do not automatically imply new Shannon information or independent model capacity. A fixed invertible Hadamard chart may be folded into a standard head for inference; exp feature gates are an established nonlinearity.
- Task overlap: 10 experimental fresh partitions are **not 10 independent datasets**, and the two sets are sequential hypothesis discovery/replication, so do not pool them into a fake first-time trial. Report paired seed differences rather than treating 5×450 image labels as independent observations.
- Real hardware caveat: this is eager CPU PyTorch 2.10 FP32 with unlocked clocks, batch128, warmup20 and 100 timed calls. No CUDA/GPU transfer/VRAM/native paper throughput extrapolation. Full file bytes are physically serialized NPZ; packed/fused/compiled code could alter runtime ranking.
- The full source was frozen on GitHub before MAT03R fresh. Six repeat train/eval cells (3 native vs Mirror/linear for seed301, three for seed401) replayed with **zero difference in deterministic numeric fields**. Ten original MATLAB-like mechanism unit tests pass; historical MAT01/MAT02 tests remain intact.
- U: five overlapping data splits, optimizer stochasticity fixed, early-phase source code freeze distinction, possible repeated-dataset dependence, surrogate physical latency, numerical precision, task thresholds estimated from training data. No coverage-calibrated population 95% CI from n=5. For task loss L in nat/label, \(u_c^2=u_{seed}^2+u_{test}^2+u_{numeric}^2\) only under independence; include covariance otherwise. k=2 expanded uncertainty is indicative rather than guaranteed.

## Reproduce

- [Pre-audit protocol for initial study](../MAT03_FROZEN_PROTOCOL.json), [source-freeze and new-seed replication protocol](MAT03R_FROZEN_PROTOCOL.json).
- [Frozen PyTorch implementation](source/run_mat03.py), [ten tests](source/test_mat03.py), [repetition runner](source/run_mat03_replica.py).
- [Initial dev/fresh and independent replication CSVs](results/), [compact results](results/RESULTS_CORE.csv), [hashes and numerical gates](results/VERIFICATION.json), [six-cell exact replay](results/REPLAY_VERIFICATION.json).
- No main, worker queue, other MA status or original science claim was edited by these experiments.
