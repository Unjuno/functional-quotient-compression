# MA-887 — exact position View and approximate cross-model translation

## Decision

**FAIL for Mirror-specific systems value; exact algebraic View passes its numerical gate.** Pythia-70M global RoPE position shift is reproduced from a canonical K/V cache with bounded FP32 differences and an 82 B phase payload. This is the known deferred-position algebra in PA156, not a new Mirror-specific operation, and the unfused materializing implementation is slower than native shifted prefill. The separate Pythia70→SmolLM2 ridge control remains approximate and misses the +1 nat quality gate in all three fresh worlds.

## H — hypothesis

A RoPE-only same-model cache has an exact global position-shift View represented by a tiny phase code, while a separately trained SmolLM2 target is not functionally equivalent and requires an approximate learned translator with observable quality error. Exact and approximate cases should have sharply different error and state costs.

## T — experiment

Pythia-70M was tested at absolute position offsets 0 and Δ=17. The exact path removes source RoPE from cached keys, restores the shifted phase, and replays the same query at shifted position IDs. The reference is a native full prefill at positions shifted by 17. The approximate control fits independent per-role affine ridge maps from Pythia-70M to SmolLM2 with nearest-relative layer/head routing and source/target RoPE normalization.

Development used bucket 0 (24 fit / 8 check / 16 test); fresh seeds 11887/11891/11897 used disjoint article buckets 1/2/3. Prompts are Wikitext-103 train articles excluded by hash from MA-883 and MA-886; Wikitext-2 rows/titles were excluded. Source/target tokenizer offsets match exactly. CPU only; no GPU or fused lazy-attention kernel. Fresh payload and model hashes are stored in the machine-readable files. Full-ridge fitting took 0.889s across the three fresh buckets; maximum RSS was 2820 MiB.

## D — result

Status: **FAIL overall; exact-view subclaim passes**.

**Facts — exact same-model view.** Across all fresh prompts, maximum absolute logit difference from shifted native replay was 0.009033; maximum NLL difference was 0.002846 nat, within the frozen FP32 tolerances 0.02 and 0.005. Mean phase-view NLL by seed was 11887=3.9936, 11891=4.2786, 11897=4.5136; native shifted NLL was 11887=3.9936, 11891=4.2788, 11897=4.5133. Serialized phase payload is 82 B; materialized shifted K cache averages 1,089,536 B across fresh prompts (seed means 936,960–1,171,968 B); phase metadata is 0.0075% of that average K-cache size. The phase path took 11887 0.080s, 11891 0.079s, 11897 0.071s including source prefill, materialization, and query replay; native shifted prefill took 11887 0.071s, 11891 0.057s, 11897 0.052s. This implementation was slower in all three buckets.

**Facts — approximate cross-model view.** Mean native SmolLM2 NLL was 3.169; independent ridge NLL was 4.795 nat, a mean increase of 1.626 nat. Per-seed increase was 11887=1.447, 11891=1.014, 11897=2.417 nat, so every fresh world misses +1 nat. Ridge payload is 2,995,321 B versus 82 B for the phase coordinate. Decoding serialized ridge payloads reproduces NLL exactly (max delta 0.0 nat).

**Interpretation.** Known positional symmetry permits a tiny functional address while preserving the same-model next-token function to FP32 tolerance. That supports an exact algebra route for compatible cache views. Separately trained cross-model states do not inherit this identity: the ordinary ridge maps retain measurable target quality loss. The measured CPU implementation pays transform and query work and does not improve latency. Exactness here aliases known LazyAttention/deferred-position algebra; no Mirror-specific advantage was established.

## C — strongest counter-hypothesis

The strict exactness result depends on one global RoPE shift in a RoPE-only architecture and one-token replay. Relative position invariance may not hold with absolute embeddings, nonlinear position scaling, sliding-window resets, or multi-token continuation. CPU timing is sensitive to unfused Python tensor layout and does not establish GPU lazy-kernel cost. The approximate mapper uses only 24 calibration prompts and unweighted ridge, not CacheBridge attention weighting or target context correction.

## U — unresolved

No multi-token continuation, long context, absolute-position architecture, RoPE scaling variant, zero-copy fused kernel, GPU throughput, or additional model pair was tested. No Mirror-specific capacity or cross-model equivalence claim follows.

## Fact / interpretation / hypothesis separation

- **Fact:** per-seed exactness errors, approximate NLL, payload sizes/hashes and wall times are in `fresh_results.json` and `RESULTS_CORE.csv`.
- **Interpretation:** exact algebraic reuse is structurally distinct from approximate cross-model mapping; the known exact case is not a Mirror-specific win and this unfused implementation is slower.
- **Hypothesis:** other architecture-preserving cache symmetries may give exact Views; broader target maps may need attention-weighted or nonlinear controls. Untested.
