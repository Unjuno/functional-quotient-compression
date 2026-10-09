# Function-first shared/private nonlinear FFN follow-up — 2026-10-09

**Scope:** CPU-only source-controlled handwritten-digit MLP experiment. **FF-NL-004/005/006: precommitted development FAIL. FF-NL-007: PROMISING on 6 new development splits and PROMISING_FRESH_REPLICATED_NARROW on 3 previously unopened fresh splits.** Not an official MA status; no global Mirror-specific advantage, no novel Mirror transform, and no LLM compression result.

## Research question

Given 12 independently trained nonlinear FFN function deltas, can a shared low-rank core plus a tiny private task residual retain new task quality at reduced *actual inference payload bytes*, beyond an ordinary LoRA rank4 initialized with **the same source-only information**? Prior FF-NL-001–003 established that a native source-warm LoRA is a strong direct comparator and that simple low-dimensional quadratic codes fail.

Dataset: 1797 real scikit-learn 8x8 digit images, artificial domain rotations/blurs/shifts/occlusion. Architecture 64→96→64→10, with a frozen clean-trained trunk and independent trained source FFN internal W2 deltas. Discover U/V on source training examples only; fit each target task from 192 permitted labeled images. Target domain names are frozen and never used to fit the source basis. All methods share training minibatch streams and update opportunities; only the target architecture differs.

## The distinct experiments

| Study | Source-controlled mechanism | Development observation | Verdict |
|---|---|---|---|
| FF-NL-004 | 8-dimensional source-PCA linear code vs matched 8-dimensional noncommutative quadratic coordinate vs source-warm LoRA4 | Nonlinear code acc 59.51%, CE 2.0459, full 8-task payload 69,474 B; source-warm native LoRA4 acc 81.49%, CE 0.5818, 80,436 B | FAIL |
| FF-NL-005 | Shared source-trained rank10 core plus task-private rank1/2 | rank1 acc 81.34%, CE 0.5740, 77,830 B vs native warm LoRA4 acc 82.36%, CE 0.5301, 80,436 B | FAIL strict dev gate |
| FF-NL-006 | Continuous target training at 180 and 540 updates from identical optimizer streams | At 540: rank1 acc 84.96%, CE 0.4774 vs native warm LoRA4 acc 85.46%, CE 0.4660; one seed missed the frozen accuracy margin by about 0.045 percentage points | FAIL strict dev gate |
| FF-NL-007 dev | 6 newly drawn data splits (6901–6906), same 540 updates | rank1 acc 85.86%, CE 0.4876 vs native acc 86.36%, CE 0.4835; 8-task payload 77,830B vs 80,436B | PROMISING_DEVELOPMENT_ONLY |
| FF-NL-007 fresh | 3 unopened seed splits (6911–6913); original science code and all training settings identical, fresh decision threshold frozen before opening | rank1 acc 84.95%, CE 0.4988 vs native acc 85.55%, CE 0.5219; candidate CE better in all three seeds and full inference bytes 3.24% lower for K=8 | PROMISING_FRESH_REPLICATED_NARROW |

**FF-NL-007 development:** seed-paired CE gap (candidate - native) mean +0.004184 nat/example, exploratory split-conditional 90% bootstrap interval [-0.008894,+0.016356]; top1 accuracy mean gap -0.493 percentage points, interval [-0.963,+0.093]. Predeclared thresholds pass. Because all seeds split the same underlying digit dataset, these are not population inference confidence intervals.

**Fresh confirmation:** 3 fresh seeds: CE gap mean -0.023170 nat/example; top1 accuracy gap mean -0.604 percentage points. Individual fresh seed6913 has accuracy gap about -1.88 percentage points, within the separately frozen -2 point safety boundary. **No retrospective adjustment of the earlier 004–006 FAILs or the original 007 dev/fresh contracts.**

## Actual bank size crossover, using individually fitted task codes

At K=1: shared/private rank1 63,998B versus native warm LoRA4 57,728B (10.86% larger).
K=4: 69,926B versus 67,460B (3.66% larger).
K=6: 73,878B versus 73,948B (0.09% smaller).
K=8: 77,830B versus 80,436B (**3.24% smaller**).

All numbers are physically serialized lossless NPZ whole-bank payloads, including the common trunk, paid source-only U/V and mean core, task-specific core+private factors, and reconstruction metadata. Startup/generation costs and offline training overhead are reported separately. K<8 uses actual trained task-code prefixes, not combinatorial or fabricated new functions.

The architecture is an ordinary shared-factor/private-residual model:
`W2_task = W2 + U (Cbar + Ctask) V.T + B_private @ A_private`.
U is (64,10), V is (96,10), C is (10,10), B is (64,1), A is (1,96), so both added terms are (64,96). Native shared-private factorization / CtS-like references predate this project; the important contribution is the measured bounded Pareto point, NOT naming it a novel Mirror operator.

## Test conditions and verification

- AMD EPYC 9V74, snapshot 2.596 GHz not locked, CPU quota 4 cores, 4 GiB cgroup memory cap, PyTorch 2.10.0+cpu FP32 eager, one intra-op thread; no GPU.
- Per-seed clean pretraining 420 optimizer steps; 12 source independent task deltas ×210 steps; shared source U/V supervised fit 240 steps; 192 labeled target examples and paired 180/540 updates. Offline source costs are NOT zero.
- Folded inference batch1 and64 were within frozen +25% time gate relative to warm LoRA in the six-seed dev sample and fresh sample, but eager materialization can be slower. CPU clock/scheduling noise limits speed conclusions.
- **28/28 tests passed. 944 recorded metric rows and 236 actual NPZ inference-bank payloads were replayed from saved weights and SHA-size checked.** Maximum absolute CE replay discrepancy <1e-6 nat/example; accuracy exactly matched.
- Six-development-seed batch execution was interrupted after seed6903, then resumed per seed using a wrapper without changing the hash-locked science module, protocol, task identities, or thresholds. All **48** initial NPZ payload hashes from 6901–6903 matched exact reruns. The interruption is not evidence for a scientific outcome.
- 007 source SHA256 `244d7aecb349d70fbc4787a5b08732f391b9bd71106996d292f9266065febea7`; dev protocol SHA256 `6afdb97781d2718388548c50e8f5de341069ad2dd1b38ce3a28deef35fc561be`; frozen fresh protocol SHA256 `250250d782d73443e23bad8a00830954f1aad4ba7a8f6e83dcacde0bd653ee76`.
- The *complete* source/model/CSV/test/log evidence archive was generated on the same conversation surface: `FFNL_2026-10-09_FULL_EVIDENCE.zip` (SHA256 `bb7ea18f9fb3d8835dab6dd2b43ee00d58bd5d5ff4465fb2be9ca4d13d892ea2`), and source/report-only archive `FFNL_2026-10-09_SOURCE_REPORTS.zip` (SHA256 `ef8b95bec48e59033911a5c7103b901cc164815210fafb991795c9d06edd27e4`). **The ZIP contents were not pushed to GitHub; this branch stores the scientific overview/provenance only.**

## Interpretation and next test

The useful empirical finding is a small, n=8 shared/private physical-storage advantage at approximately matched strong source-warm LoRA quality in a small CPU classification bank. It is *not* an asymptotic capacity, 64×, natural-language, KV-cache, or Mirror-specific theorem; it does not justify changing official MA statuses. One natural disjoint task-family or larger nonlinear multi-layer bank with a fully budget-matched CtS/BOLT/native-LoRA control is the next essential falsification, including basis-training amortization and optimized-kernel throughput.
