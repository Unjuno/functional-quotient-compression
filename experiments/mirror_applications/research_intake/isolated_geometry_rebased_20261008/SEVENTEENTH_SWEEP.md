# Seventeenth literature sweep — strong native controls for Mirror

**Date:** 2026-10-08 JST. **Branch:** `research/mirror-isolated-protocols-rebased-20261008` (research-isolated). **Status:** 6 new UNTESTED MA plans, 9 new PA references, 3 existing-ID control supplements; no new trained-model success claims.

## Verified source map

| PA | Original paper | Native protection against false Mirror success |
|---|---|---|
| PA425 | [xKV — ICML 2026](https://proceedings.mlr.press/v306/chang26d.html) | Shared cross-layer basis and selective KV reconstruction are already native |
| PA426 | [KQ-SVD — AISTATS 2026](https://proceedings.mlr.press/v300/lesens26a.html) | Query-aware score reconstruction is native, not Mirror |
| PA427 | [ALoRA — ACL 2026 Findings](https://aclanthology.org/2026.findings-acl.625/) | Sharing output B rather than input A is already established |
| PA428 | [mtLoRA — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/791de7c35bb49cfca56744e67f90eef4-Abstract-Conference.html) | Spectral-aware block-level routing has a native implementation |
| PA429 | [FACET — arXiv 2026](https://arxiv.org/abs/2608.31096) | One shared adapter and dynamic task transform are native |
| PA430 | [MoLoRA — arXiv 2026](https://arxiv.org/abs/2603.15965) | Per-token LoRA routing is native |
| PA431 | [NeuroLoRA — arXiv 2026](https://arxiv.org/abs/2603.12378) | Dynamic context gating on sparse features is native |
| PA432 | [Quantization Dominates Rank Reduction — arXiv 2026](https://arxiv.org/abs/2604.11501) | True bitpacked INT4 can defeat rank reduction at equal memory |
| PA433 | [Shared-Prefix KV Reuse Standard LoRA — arXiv 2026](https://arxiv.org/abs/2609.17109) | Numerically reused KV may physically copy buffers or hurt expert quality |

**Evidence types:** PA425/426/427/428 have conference venue confirmation; PA429/430/431/432/433 are research preprints, to be evaluated with stricter reproduction caveats.

## Full worker-free experiment plans

| New ID | Distinct incremental m hypothesis | Strongest native prior | Priority |
|---|---|---|---|
| [MA-1165](plans/MA-1165/README.md) | Further compress xKV reconstruction factors beyond xKV-SR | PA425 | P0 |
| [MA-1166](plans/MA-1166/README.md) | Amortize multiple role-specific KQ-SVD projector banks into small m | PA426 | P0 |
| [MA-1167](plans/MA-1167/README.md) | Compress mtLoRA's fine-grained task routing beyond simple SVD | PA428 | P0 |
| [MA-1168](plans/MA-1168/README.md) | Reduce native FACET task-state without replay or task-ID oracle | PA429 | P1 |
| [MA-1169](plans/MA-1169/README.md) | Replace physical MoLoRA specialist bank while maintaining cache correctness | PA430 | P0 |
| [MA-1170](plans/MA-1170/README.md) | Add functionally useful m beyond native NeuroLoRA contextual gate | PA431 | P1 |

Every folder includes `README.md` (H/T/D/C/U, shapes/variables/units, native controls, validation/dataset steps, failure analysis, measurement contract), `PROTOCOL.json` (seed firewall and predeclared gates) and `STATUS.md` (UNTESTED). The plans include enough method structure for a **synthetic mechanism test without rereading every paper**; paper reproduction claims still require checking the author source, versions and exact method.

## Deduplicated existing hypotheses

- [MA-1097 ALoRA shared-B control](existing/MA-1097-ALORA_CONTROL.md) instead of a duplicate new A/B-sharing MA.
- [MA-578 genuine bitpacked INT4 vs Mirror rank](existing/MA-578-KV_INT4_CONTROL.md) instead of a duplicate generic quantization MA.
- [MA-1112 standard-LoRA cache reuse + physical alias](existing/MA-1112-STANDARD_LORA_PREFIX.md) instead of a new generic cache-switch MA.

## Global research protocol

1. Preserve the native algorithm, paper/generation status, base checkpoint, dtype and tokenizer.
2. Add m at the minimal interface, never rebrand native design.
3. Compare native / simplest same-byte linear or diagonal control / Mirror / independent upper bound.
4. Separate exact algebra, aligned synthetic, independent natural task transfer, actual bytes and real runtime into different evidence lanes.
5. Freeze development seeds 11–13 and fresh 101–105 on entire task identities. Require n_min=5 (early screen) and >=10 independent replication worlds before ADOPTED.
6. Compute paired task-world CIs and `u_c` with covariance where relevant; k=2 expanded U is only indicative at five worlds.
7. Actual serialized inference bytes (all bases, private exceptions, code, cache, metadata, router) are authoritative. CPU eager proxy is not end-to-end GPU kernel evidence.
8. A negative result is retained; even native-method parity implies M0, not a Mirror-specific breakthrough.

## Isolation instructions

The **canonical worker branch** `research/mirror-application-worker-ready-20261007`, current next candidate `MA-255`, WORKER_START_HERE, CONTEXT_ROUTER, FIRST_QUEUE, WORKER_QUEUE and main are not modified. This document lives **only** on `research/mirror-isolated-protocols-rebased-20261008`. Do not auto-claim new MA IDs, do not preempt current worker queue, and re-check canonical max MA and PA IDs before promoting any plan.
