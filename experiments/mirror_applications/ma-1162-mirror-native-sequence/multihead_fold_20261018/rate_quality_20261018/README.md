# MA-1162: minimum-rate physical QKV + multihead Mirror code

**Date:** 2026-10-18. **Status:** follow-up rate/quality compression screening (not natural-language or standalone MHA deployment).

> **Mirror insertion:** replace per-head independent Q and output-side View matrices with compact Q/output coordinate codes acting on ONE shared physical QKV bank, while maintaining multiple logical causal Attention heads without storing multiple prefix K/V copies.

The core question is not whether logical head count increases. It is whether the **serialized inference payload (shared coefficients + View codes + header + any paid model-specific metadata) and the resident physical KV cache** are minimized **subject to retained head function**. The cost of the generic executable program is reported separately rather than hidden. The physical base QKV here consists of 384 stored FP16/FP32 elements and is **frozen during the code-only teacher fit**: this is an isolated mechanism experiment, not a whole trained Transformer.

Tests: deterministic bit-packing/roundtrip; actual payload bytes; functional deviation against aligned linear/nonlinear teachers and a deliberately off-orbit teacher; head KL; per-head native folded matrix size. FP16/FP32 and 8/12/16/32-bit View codes. Initial development world was 16110, **fresh worlds 16111..16113 fixed before opening**. Details: `PROTOCOL_RATE_QUALITY_FROZEN.json`.

The **native exact-function linear folding control** stores per-head Q,V projections plus common K, and represents the *same functions* as static linear Mirror; it is not a claim that dense unrelated heads could always be compressed. The independently parameterized QKV matrix bank is a separate quality upper-bound reference, not equal capacity. For nonlinear post-attention activations, a naive static linear projection fold is mathematically invalid.

Basic read-only replay:

```bash
cd experiments/mirror_applications/ma-1162-mirror-native-sequence/multihead_fold_20261018
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q rate_quality_20261018/tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python rate_quality_20261018/source/rate_quality.py --phase dev --out /tmp/ma1162-rate-dev.csv
```

Fresh reproduction uses `--phase fresh` with a **new** output path. `--phase` prevents accidental use of old numeric worlds; all CSV output refuses to overwrite.
