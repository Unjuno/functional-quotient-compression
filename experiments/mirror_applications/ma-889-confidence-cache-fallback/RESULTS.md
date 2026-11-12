# MA-889 — confidence-gated cross-model KV translation

## Decision

**FAIL.** The source-only cache RMS gate preserved native quality for the Pythia-160M pair by falling back on every prompt, so it violated the 50% fallback ceiling. For SmolLM2 the frozen threshold translated most prompts, but mean fresh NLL regret was **+1.277 nat** and exceeded the +0.5 gate in two of three seeds. The combined fanout path was faster than two native target prefills in all three fresh buckets, and gate state added only 30 serialized bytes to the full per-pair ridge maps. The reliability/quality gates failed.

## H — hypothesis

A source-only cache novelty score, calibrated on development data and read without target loss/logits at inference, would route unreliable translated KV states to native target prefill. It would keep each target within +0.5 nat, use native fallback on at most half of prompts, and improve two-target fanout latency.

## T — experiment

Source: EleutherAI/Pythia-70M-deduped. Targets: EleutherAI/Pythia-160M-deduped and HuggingFaceTB/SmolLM2-135M. Each pair used a separately fitted full per-role affine ridge KV mapper (alpha 0.01), nearest-relative layer/head alignment, inverse RoPE for fitting, and target RoPE restoration. The source-only gate score was source KV RMS standardized against the pair's 24 map-fit prompts. Development bucket 0 selected thresholds Pythia-160M `-inf` (always fallback; no threshold met both check constraints) and SmolLM2 `1.0`. Thresholds were frozen before fresh scoring.

Fresh seeds 12889/12893/12899 used three whole-article Wikitext-103 train buckets (24 fit / 8 unused check / 16 test per bucket). Token offsets/counts were aligned across all tokenizers. The manifest excludes Wikitext-2 and selected MA-883/886/887 hashes; row overlap with those experiments is zero. No optimizer steps were used; maps were closed-form fits. CPU only, CUDA unavailable. Fresh cache preparation took 22.4/25.7/22.8 seconds; peak RSS was 3,249/3,771/3,886 MiB. Model hashes and checkpoint bytes are in `fresh_results.json` and `PAYLOAD_INVENTORY.json`.

### Frozen controls

- Native target prefill and one-token replay.
- Always translate with independent per-pair full ridge maps.
- Source-only RMS gate with native prefill fallback.
- MA-886 pooled-map results are historical context only.

The gate never reads target cache, logits, labels, or loss. The measured fanout comparison charges one source prefill shared by both target handoffs and then each target's translated replay or native prefill/replay.

Always-translate full ridge mean NLLs by seed were Pythia-160M 3.927 / 4.450 / 5.078 nat and SmolLM2 4.472 / 3.260 / 4.197 nat. The gate improved SmolLM2 over always-translate modestly on each bucket, but remained outside the preregistered mean-regret gate overall.

## D — result

### Facts

| Target | Fresh regret vs native (seeds 12889 / 12893 / 12899) | Fallback fraction | Result |
|---|---:|---:|---|
| Pythia-160M | 0.000 / 0.000 / 0.000 nat | 100% / 100% / 100% | Fails fallback ceiling |
| SmolLM2 | +1.847 / +0.391 / +1.594 nat | 6.25% / 12.5% / 12.5% | Fails +0.5 nat quality gate in 2/3 seeds; mean regret +1.277 nat |

Combined two-target fanout latency, with one shared source prefill, was **0.348 / 0.461 / 0.329 s** versus **0.365 / 0.525 / 0.380 s** for two native target-prefill paths. It was faster in 3/3 buckets, but this does not rescue quality/fallback failure.

Each serialized payload was **7,787,782 B**, containing the full per-role ridge maps plus pair IDs, two RMS normalizers and two thresholds. The same ridge-map payload without gate state was **7,787,752 B**; the gate adds 30 B (0.00039%). This is reliability metadata over a full mapper, not map compression. For context, the producer checkpoint is 166,029,852 B and the two target checkpoints total 644,059,248 B; a service holding all three adds the producer to a target-only 644,059,248 B system.

Serialized map roundtrip changed NLL by exactly 0.0 in every target/bucket. Post-score artifact audit found the copied serializer's five-byte header still said `MA886`; the header was corrected to `MA889` and payloads rehashed. Only the format label changed; mapper/gate arrays, thresholds, metric values and byte counts did not. Original pre-fix and corrected hashes are preserved in result JSON. Six implementation/provenance tests pass.

### Interpretation

Source-cache RMS magnitude did not provide a useful reliability boundary across these model pairs. Pythia-160M could only satisfy development quality by reverting on every prompt, while the SmolLM2 threshold that looked acceptable on the development check failed on two fresh article buckets. The latency reduction is real for this one-token CPU fanout harness, but it comes with failed quality or fallback requirements. The gate adds negligible bytes, while the full maps remain 7.79 MB and unchanged.

## C — strongest counter-hypothesis

The dominant uncertainty may be pair- and role-specific representational mismatch rather than source-state magnitude. A calibrated target-side context correction or a richer source-only feature could separate safe from unsafe translations; neither was tested here.

## U — unresolved

No multi-token rollout, longer context, target-side correction, CacheBridge/MoT reproduction, GPU/fused path, or additional model family was evaluated. The result does not establish that all source-only reliability signals fail; it falsifies this frozen RMS gate on these two pairs and buckets.

## Fact / interpretation / hypothesis

- **Fact:** fresh NLL, fallback fractions, latency, payload bytes/hashes, checkpoint hashes/sizes and split provenance are stored in machine-readable artifacts.
- **Interpretation:** RMS-only gating is not reliable enough for these pairs under the registered quality/fallback gates.
- **Hypothesis:** richer source or target-side reliability signals might identify failures at lower fallback cost; untested.
