# MA-1162 — rate/quality screening of compact Mirror multi-head Attention (2026-10-18)

**MECHANISM PASS for aligned synthetic heads; off-orbit FAIL; real language/normal MHA compression UNTESTED.**

This is a follow-up to the pre-fresh protocol and code already committed under this folder. Historical MA-1162/691/253/1164 statuses are not modified.

## Goal and implementation

One physical Q/K/V tensor bank (3 matrices each 16×8; 384 elements) and eight head-specific scalar codes (two per logical head) implement **four logical causal Attention heads**. No virtual Q/K/V matrices are stored in the compact model payload. Physical QKV weights are frozen buffers during the head-code fitting. Nonlinear View includes tanh on query/output after attention. The assumption of one shared KV producer is mandatory.

Custom payload format: 14-byte metadata header, little-endian FP16/FP32 QKV bank, bit-packed periodic Givens codes with 8/12/16/32 bits per angle. The codec source and generic runtime are external dependencies and must be charged or amortized separately. All packed payloads were decoded and actually evaluated with PyTorch FP32.

## Preregistered follow-up procedure

- Development: world **16110** selected the smallest among 8/12/16/32 code precision variants meeting aligned teacher MSE < 1e-5 on both linear and nonlinear families.
- New fresh: **16111, 16112, 16113**, two initializations (1,2), 160 Adam steps at LR .06, batch32, 256 evaluation examples per run.
- Three families: aligned linear, aligned nonlinear, **off-orbit independent dense orthogonal heads**.
- 144 evaluations: 3 families ×3 worlds ×2 seeds ×2 physical precisions ×4 code precisions. No dropped rows.
- 24 unit tests PASS (including 9 original MA-1162 head tests and 15 new code/roundtrip tests).

## Storage versus quality

| Representation | Raw same-header FP16 inference-model payload | Applicability |
|---|---:|---|
| Unrestricted independent QKV per head | 3,086 bytes | parameter-budget/quality reference, not fitted here |
| Native independently folded per-head Q/V plus shared K | 2,318 bytes | EXACT for linear View, not nonlinear tanh View |
| Shared physical QKV without View codes | 782 bytes | cannot express four distinct head functions in general |
| **Shared physical QKV + eight 12-bit Mirror codes** | **794 bytes** | matched aligned teacher; 6/6 per family |

794 bytes = 768 bytes shared FP16 QKV + 12 bytes angle codes + 14 bytes codec metadata; ~74.3% below raw fully-independent QKV payload and ~65.7% below raw linear-folded Q/V payload. The 12-bit solution is minimum **only among the tested bit widths**. It is not a global compression optimum.

| Teacher | Mean held-out MSE at FP16 + 12-bit codes | Worst of 6 MSE | Smallest pairwise head-attention KL |
|---|---:|---:|---:|
| Aligned linear | 1.3633e-7 | 1.5254e-7 | 0.19558 |
| Aligned nonlinear | 2.3183e-8 | 2.7388e-8 | 0.004838 |
| Off-orbit independent dense | 0.79456 | 0.85112 | 0.007236 |

Preregistered quality gate MSE<1e-5 and KL>1e-4 passed 6/6 for each aligned family. Off-orbit failure is preserved, not excluded.

**Memory note:** at prefix 8192, batch1, physical 1 KV head × dimension8, hypothetical FP16 cache is 256KiB. Four materialized logical KV views occupy 1MiB. These are tensor-size calculations only; this follow-up did NOT measure new physical allocator aliasing, live peak VRAM or natural LM cache quality. Earlier MA-691 contains a separate exact-cache mechanism audit.

**Runtime note:** fixed linear View can be absorbed into native Query/output projections and can be faster; no Mirror-specific inference speedup or trained language-model gain is claimed. For this tiny pilot, separate Python decoder/runtime source files are much larger than the 794-byte model payload; per-model cost advantage requires reusable compiled runtime and a fair native implementation baseline.

## Evidence and reproducibility

GitHub contains the exact preregistered code and protocol in this folder. This follow-up's full fresh CSV, development CSV, SHA-256 verification JSON, and Japanese scientific report are supplied in the current research conversation archive. Files to recover:

- `results/fresh.csv` — SHA-256 `7d823f75263f84b19132c30ec14a887c5479d5ce30378ce2db655c45854f1ddf`
- `results/VERIFICATION_METRICS.json` — SHA-256 `19bacc62c54cd45863cf84763ef827242ad592d86e67e8dbe6ed57f93496aab5`
- `REPORT_JA.md` — SHA-256 `356ff842d7f9e02af87551b4a02d97c4949133461624d3e98c6beeea5e050113`
- `VERIFICATION.json` — SHA-256 `a7c0054b9a371b1285867493793e9c0fe3c5901386c56147c253e8134b109fb1`

Fresh results were not used to tune the previously frozen source or decision thresholds. The existing earlier multihead benchmark has one FP32 numeric 2e-6 threshold miss in 24 configurations; it is not retroactively rewritten.

## Decision

**Aligned multi-head code storage compression: PASS (limited synthetic mechanism). General MHA/Transformer quality and GPU/speed win: UNTESTED.**

Next: learned shared physical QKV and output mixer in a causal LM, compared at the same actual bytes with GQA/MQA/MLA, per-head low-rank/gain, and algebraically folded native; count code/runtime/cache and end-to-end throughput.
