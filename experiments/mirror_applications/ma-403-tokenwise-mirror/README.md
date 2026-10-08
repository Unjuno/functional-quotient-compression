# MA-403 — Token-wise Mirror modulation: CPU screen

2026-10-09 JST. **Development verdict: FAIL. Fresh seeds remain sealed.**

## Insertion and scope

A causal-prefix-generated angle code rotates consecutive residual-channel pairs immediately before the FFN of one shared frozen causal Transformer block. No teacher angles, task IDs, or generator parameters are given to the learner. The targets are synthetic teacher categorical distributions: this is not natural-language modeling, pretrained-model adaptation, a capacity proof, or autoregressive multi-token generation.

Controls: no modulation, static Mirror, local-token Mirror, causal-prefix Mirror, FiLM, unconstrained complex coefficients, pairwise 2x2 residual matrices, rank-2 dynamic residuals, and a full conditional matrix. All generated conditioners use an eight-unit tanh hidden layer. Pair rotation and equivalent complex-phase expressions are not counted as new independent functions.

Direct prior-art references: PA63, Perez et al., FiLM, arXiv:1709.07871v2; PA64, Park et al., SPADE, arXiv:1903.07291. SPADE motivates position-specific conditioning, but image synthesis is not reproduced here.

## Frozen experiment

Vocabulary 24; one frozen random causal block; width 24; four heads; FFN 48; sequence length 12; FP32; dropout zero. Three controlled teachers: rotation, affine modulation, dense mixing. Development seeds 40301/40302. Nine methods produce **54 configurations**, with 48 learned models trained for 800 updates (batch 32, AdamW, initial learning rate 0.008, cosine decay). Total primary updates: 38,400.

Each world: 1024 training sequences, 256 IID sequences, 512 OOD sequences. Training excludes adjacent modulo-four groups (0,1) and (2,3); OOD metrics score those transitions. Frozen-prefix caching is permitted equally during training; inference timing includes the complete prefix computation.

## Quality

OOD excess KL, nat/token; ranges across two development seeds, not confidence intervals:

| Teacher | Causal Mirror | Representative native control | Full-matrix reference |
|---|---:|---:|---:|
| Rotation | 0.003427–0.005402 | complex coefficients 0.025159–0.030310 | 0.031202–0.031351 |
| Affine | 0.153820–0.222510 | FiLM 0.000645–0.001002 | 0.006664–0.007296 |
| Dense | 0.199265–0.237671 | pairwise matrices 0.163945–0.174688 | 0.003161–0.003979 |

On rotation worlds, static Mirror gives 0.261567–0.364328 and local-token Mirror 0.237187–0.314666. Context-dependent conditioning is useful in that aligned family, but does not transfer to the off-family conditions.

The frozen gate requires Mirror within 0.01 nat/token of the best native control, absolute KL at most 0.03, improvement over static Mirror at least 0.005, no increase in full bytes, at most 80% incremental bytes of the smallest quality-qualified native, and at most 1.25x latency of the fastest quality-qualified native at batches 1 and 64. The full-matrix reference must also reach KL at most 0.03.

Both rotation cases pass Mirror's quality, dynamic usefulness, storage and runtime gates, but their full-matrix reference fails the OOD threshold: formal verdict **NOT_ESTABLISHED**. Four affine/dense cases **FAIL** with learnable full-matrix references. Overall **FAIL**; no gate changes or fresh-world evaluation.

## Actual storage and CPU latency

All inference bytes include frozen backbone, generator, adapter, dtype, shapes and reconstruction metadata. No quantization.

| Method | Full bytes | Incremental bytes | Trainable adapter parameters |
|---|---:|---:|---:|
| No modulation | 25561 | 0 | 0 |
| Static Mirror | 25695 | 134 | 12 |
| Causal Mirror | 27156 | 1595 | 308 |
| Complex coefficients | 27595 | 2034 | 416 |
| FiLM | 28452 | 2891 | 632 |
| Rank-2 | 27335 | 1774 | 314 |
| Full matrix | 47477 | 21916 | 5384 |

Mirror saves 21.6% of incremental bytes against the compact rotation-family control, but only **1.59% of full-model bytes**. Do not interchange these numbers.

AMD EPYC 9V74; clock snapshot about 2.596 GHz, not locked; five visible CPUs, four-core-equivalent quota; Torch one intra/inter-op thread; cgroup memory 4 GiB. Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5; CPU eager FP32 causal SDPA, no compilation. Primary peak RSS 411000 KiB.

Rotation-family Mirror full-sequence latency medians: 0.2791–0.2907 ms at batch 1; 1.2101–1.3344 ms at batch 64; 12 input tokens. This is 5.9–13.9% and 1.5–1.7% slower, respectively, than the fastest quality-qualified native control. It meets a tolerance, not a speedup. Seven interleaved repetitions, ten calls per repetition, warmup 15 calls; 756 samples across the primary matrix. These are uncached sequence-forward timings, not KV-cached autoregressive decode timings. Repeat dispersion does not account for all scheduling or clock uncertainty.

## Separately scoped post-hoc diagnostic

The full-matrix rotation reference was trained from scratch for 3200 updates on the same observed development worlds. Its schedule was stretched to that budget. OOD KL became 0.023045765944042042 and 0.022182795065245675, below the original 0.03 bound. This identifies an optimization-budget contribution; it is not a fresh replication, a capacity theorem, or a promotion of the original FAIL. Same-width full-matrix conditioning is not a mathematical superset of the nonlinear angle parameterization.

## Verification

10 unit tests passed. Strict causal-prefix testing found cancellation in inclusive cumsum minus current activation; shifted cumsum corrected it before development, without relaxing exact-zero test tolerance. All 54 inference payloads reproduce 810 metric scalars exactly; both rotation Mirror trainings reproduce complete payload hashes from initialization. Two diagnostic payloads reproduce another 30 metric scalars. Latency is not bitwise reproducible. Historical repository-wide tests were not rerun.

Protocol and measured code hashes are in PRE_RUN_LOCK.json and VERIFICATION.json. Fresh seeds **40311, 40312, 40313 remain unopened**. The full conversation evidence bundle additionally retains all inference payloads, training traces and timing repetitions; Git contains source, frozen protocol, tests and core numerical evidence.

## Reproduction

From the repository root, with CPU torch/numpy/pytest installed:

```bash
D=experiments/mirror_applications/ma-403-tokenwise-mirror
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q "$D/tests"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$D/source/experiment.py" dev --out "$D/runs/reproduction"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$D/source/experiment.py" replay --out "$D/runs/reproduction" --retrain
```

The optional post-hoc diagnostic is `python "$D/source/diagnose_upper.py"`. It does not open fresh seeds. Core result values are development evidence, not fresh confirmation.

## Provenance and continuation

All computation occurred in the current container. The local workspace is isolated and is not a clone; GitHub connector supplied the live registry and queue. Base commit: 5ffe49d3207694000e45f6dc22005b63ff871db7. Research branch: research/ma-403-tokenwise-mirror-container-20261009. Main is untouched. STATUS_DELTA.csv records only the proposed MA-403 UNTESTED-to-FAIL update; global historical statuses have not been overwritten.

Next queued P0: MA-405, subject to checking live branches. Affine-plus-rotation or private-residual extensions need separately frozen protocols; they cannot retroactively rescue this gate.

Related disciplines: numerical linear algebra (orientation versus scale), experimental statistics (aligned/off-family controls), systems engineering (real serialized bytes and timing), and conditional learning (useful causal coordinates without independent-capacity claims).
