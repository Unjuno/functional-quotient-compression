# MA-876 results: shared basis for cross-model KV translators

## H — Hypothesis

A shared low-rank basis over calibrated per-head affine KV maps can serve an unseen ordered source-target model pair with target next-token quality close to independent full ridge maps, while reducing serialized translator bytes.

## T — Trial

The screen used three locally cached pretrained models: GPT-2 (12 layers, 12 heads, head dim 64), Pythia-70M (6 layers, 8 heads, head dim 64), and SmolLM2-135M (30 layers, 3 KV heads, head dim 64). The basis was fit on GPT-2→Pythia and GPT-2→SmolLM2; Pythia→SmolLM2 was held out as an ordered pair. For each target KV head and K/V role, an affine ridge map used deterministic relative-depth/head alignment, with alpha=.01. Rank candidates were 1, 2, 4, 8, 16, and 32. Fresh data used disjoint tokenizer-offset-aligned Wikitext-2 prompts: 12 calibration prompts per pair per seed and 16 heldout test prompts per seed, for seeds 983, 997, and 1009. Native target prefill and independent full maps were controls.

CUDA was unavailable. All measurements ran on CPU; no GPU latency result is claimed. Exact tokenizer-offset matching reduced the eligible prompt pool. Fresh row IDs and model file hashes are locked in FROZEN_SETTINGS.json.

## D — Decision

**FAIL.** The 92.0% family mapper-byte saving passed the storage side, but rank32 compressed heldout-pair maps missed the frozen NLL tolerance in 3/3 fresh worlds: compressed NLL was 3.11, 3.77, and 5.34 nats above the independent full ridge mapper. Native PCA map coding is functionally the same representation and has identical bytes, so there is no Mirror-specific gain.

## C — Strongest counter-hypothesis

The heldout Pythia→SmolLM2 maps are not well represented by a basis learned from the two GPT-2-source pairs. Even independent heldout ridge maps degrade target NLL relative to native target prefill by 1.61–3.43 nats, indicating a difficult model pair before compression. The heldout calibration set is only 12 prompts per world, which may also destabilize affine maps.

## U — Unconfirmed

The result covers one heldout pair, 16 test prompts per seed, one fixed layer/head assignment rule, Wikitext-2, and CPU execution. CacheBridge-style attention-sensitive calibration, learned source-layer selection, other ordered pairs, larger calibration sets, GPU latency, and target NLL with a robust fallback remain untested.

## Facts

- Development did not find a rank meeting the <=0.05 nat NLL-delta gate. Rank32 had the smallest dev delta versus full ridge (2.883 nats); it was frozen before fresh prompt evaluation. Comparing each candidate to full ridge instead of native target NLL does not change the ordering because the full-map NLL is constant across ranks on a world.
- Fresh compressed-minus-full-ridge NLL deltas were 3.113 (seed 983), 3.767 (997), and 5.337 (1009); all failed the .05 gate. Independent full-ridge maps were themselves 1.605, 3.434, and 2.375 nats above native target prefill.
- Three-pair family translator payload was 7,590,286 B independent versus 609,934 B shared rank32, a 91.96% reduction. For the heldout pair alone, full maps were 2,996,100 B and shared-basis marginal payload was 573,060 B, an 80.87% reduction. The shared payload includes map-family mean, rank basis, all map codes, pair names, and source-layer/head metadata.
- Heldout relative cache error averaged 0.3867 for full maps and 2.969e6 for rank32 reconstruction, showing severe basis extrapolation under this mapping setup.
- The three cached model weight files total 983,195,575 B; Pythia plus SmolLM2 weights alone are 435,090,404 B. These model weights are common to all translator controls and are reported separately from marginal mapper bytes. Tokenizer/config payloads total 8,348,140 B across the three models. The 48 heldout prompts averaged 40.67 tokens (world means 36.69/41.75/43.56; range 16–115). At float32, the source Pythia KV cache is 24,576 B/token and target SmolLM2 KV cache 46,080 B/token, or about 1.00 MB and 1.87 MB respectively per average prompt.
- On CPU, mean source prefill was 36.7 ms, target native prefill 98.7 ms, map application 3.03 ms full / 2.96 ms compressed, and target suffix pass 55.4 ms compressed. Cache-map reconstruction MACs are not included in these application wall times; the rank32 decoder requires about 23.96M multiply-adds for the 180 heldout maps, followed by 748,800 MACs per translated token.
- A GPT-2 identity-cache control changed the next-token logit by at most 1.22e-4, confirming the injected-cache NLL path against native prefill. Fresh replay was deterministic.

## Interpretation

The basis achieved strong storage compression but did not generalize to the heldout pair and substantially damaged cache quality. The direct full-map control also had a material NLL gap to native target prefill, so this pair is difficult even before basis compression. The observed result is a negative cross-pair compression frontier for this basis/alignment/calibration design, not evidence against exact cache reuse or all translator families.

## Hypothesis for follow-up

A follow-up should first improve the native translator for this pair using target-attention-weighted calibration or source-layer selection, then ask whether a shared basis compresses the improved map family. Any change after these fresh results requires a new protocol amendment and new heldout prompts.
