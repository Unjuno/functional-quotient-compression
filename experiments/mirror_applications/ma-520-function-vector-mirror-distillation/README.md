# MA-520 — Function Vector to Mirror code distillation

Status: FAIL.

> **Mirror insertion:** this experiment adds a compact coordinate `m` and a shared decoder to the frozen-LM residual intervention interface so a task function vector is reconstructed from fewer paid bytes than the explicit vector bank.

PA99 motivates causal function vectors. We tested explicit prompt-delta vectors, generic rank-4 PCA coefficients, and K=8 product-quantized Mirror codes on a pinned frozen GPT-2.

## H / T / D / C / U

**H:** A compact Mirror code preserves task behavior at lower actual serialized bytes.

**T:** 16 procedure×domain prompts, 3 fresh worlds×3 seeds, explicit/PCA/PQ codes, direct ICL and query-only.

**D: FAIL.**

**Fact:** Direct ICL accuracy averaged 25%; query-only 6.25%. Explicit vectors, PCA, and PQ all scored 0%. Mean target NLL: explicit 6.4128, PCA 6.4988, PQ 6.5109. Serialized bytes per task bank: explicit 50,921B; PCA 17,505B (34.4%); PQ 14,117B (27.7%). Mean vector NRMSE was 0.1522 for PCA and 0.1755 for PQ.

**Interpretation:** PQ reduces actual payload by 72.3% versus explicit vectors, but loses task utility; it is a representation compression result only. The 20% task-utility gate failed.

**C:** Prompt-minus-query vectors do not encode executable task functions in this setting.

**U:** Causal head-specific FVs, natural tasks, runtime/decoder latency, and stronger quantizer controls.
