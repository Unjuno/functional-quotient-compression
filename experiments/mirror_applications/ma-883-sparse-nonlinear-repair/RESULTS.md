# MA-883 — sparse private nonlinear repair for cross-model KV transfer

## Decision

**FAIL** under the frozen gates. The pooled cross-model translator misses native quality on most pair×seed conditions. Every pair was marked hard during development, so the sparse-repair premise also fails. Private repair state and total payload exceed their limits. The GELU MLP is consistently worse than the matched identity-activation residual.

## H — hypothesis

A shared pooled affine KV map plus compact pair coordinates should handle easier Pythia→SmolLM2 pairs, while a small per-role GELU repair for development-identified hard pairs should recover quality within +0.5 nat of independent ridge and +1.0 nat of native, repairing at most one of two pairs, with private repair ≤20% and total payload ≤75% of the two-pair independent-ridge bank.

## T — training and evaluation

Two source pairs: Pythia-70M→SmolLM2-135M and Pythia-160M→SmolLM2-135M; 180 target K/V roles; nearest-relative-depth/head mapping and inverse/forward RoPE. Development used 48 prompts from WikiText-103-train article bucket 0: 24 fit, 8 hard-mask check, 16 width-selection/test. Fresh used disjoint whole-article buckets 1/2/3 (seeds 983/997/1009), with the same 24/8/16 split. All exact offset/token alignment checks passed. No WikiText-2 article titles/text rows were eligible.

The hard rule flagged **both** source pairs (>1 nat pair-FiLM minus native on development check), exceeding the maximum one repaired pair. Width 16 was selected by the frozen development rule. Each fresh seed fit only on its own calibration rows; all six controls were replayed on separate articles. CPU-only, three cached pretrained models. Total fresh evaluation script reported 829.1 s across seeds (plus model/data initialization); max resident set size about 2.8 GiB. Per-seed training and replay wall times are in `fresh_results.json`.

Controls: native target cache, pooled shared linear, pair FiLM, independent per-pair full ridge, identity-activation residual at matched width, sparse GELU residual, and full ridge plus GELU residual upper control.

## D — result

Status: **FAIL**.

**Facts.** Across six pair×seed test conditions, mean NLL was native 2.777, shared linear 5.172, pair FiLM 5.583, full ridge 4.889, identity residual 3.712, sparse GELU 4.149, and full nonlinear upper 3.072 nat. Sparse GELU passed the native+1 nat gate in 2/6 conditions; it was within +0.5 nat of full ridge in 6/6. It improved over pair FiLM by at least .5 nat in all six conditions, was within +.25 nat of the full nonlinear upper control in 1/6, and was worse than identity residual in all six.

**Payload facts.** Serialized sparse GELU payload: 6,522,107 B; independent two-pair full-ridge bank: 5,990,621 B (108.9%). The pair-FiLM payload was 3,089,020 B; private repair state: 3,433,087 B (57.3% of the full-ridge bank), above the 20% limit. Both pairs were repaired (100% of pairs and 100% of test tokens), violating the sparse condition. The full nonlinear upper control was 9,423,711 B.

**Serialization.** The first replay pass detected a writer/decoder repair-order mismatch; original failed deltas are preserved in `fresh_results.json`. A decoder-only correction was committed after all fresh outcomes were recorded. The post-hoc replay audit in `serialization_correction_audit.json` reproduces every stored per-pair NLL with maximum absolute difference 0.000000 nat in all 18 payload/seed rows. This repairs implementation verification only and adds no independent quality evidence. Actual payload bytes and SHA-256 hashes are listed in `PAYLOAD_INVENTORY.json`.

**Interpretation.** Pair-specific affine calibration does not create a robustly compact cross-model cache bridge at this scope. A linear private residual is more useful than the GELU nonlinearity, but the two-pair private state overwhelms the payload budget. This is a bounded failure for the two Pythia→SmolLM2 pairs and one-token replay task.

## C — strongest counter-hypothesis

Only 24 calibration prompts and 24 updates per residual may underfit the nonlinear repair; broader article coverage or longer adaptation could improve quality. That does not explain the failed hard-pair sparsity rule, the private-byte excess, or why the matched identity residual beats GELU on all six conditions. No post-fresh tuning was done.

## U — unresolved

No unseen source family, autoregressive multi-token handoff, attention-weighted cache distortion, GPU/fused runtime, energy, or native fallback policy was tested. This result is not a CacheBridge/MoT reproduction and supports no general cross-model-cache claim.

## Fact / interpretation / hypothesis separation

- **Fact:** recorded NLLs, actual payload byte counts/hashes, mask, and replay deltas are in the machine-readable artifacts.
- **Interpretation:** the specified shared/private sparse architecture misses joint quality/storage gates; ordinary identity residual outperforms GELU here.
- **Hypothesis:** more calibration or a different gated fallback could help, but remains untested and would require a new protocol/ID.
