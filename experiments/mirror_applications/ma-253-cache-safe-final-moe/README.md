# MA-253 — cache-safe final-layer Mirror-MoE

Status: FAIL for Mirror expert replacement; cache-placement mechanics PASS
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `01b515cf8d0603481cb00ff1d5be45541411eebb`

## H — hypothesis

A final-layer shared matrix plus four per-domain Givens angles can approach independent rank-2 DMoE LoRA quality with fewer serialized bytes and preserve exact prefix K/V cache reuse. Applying the same view before a later attention layer should invalidate downstream cached K/V.

## T — execution

PA14 describes DMoE's independently updatable knowledge experts and uncertainty-aware router, placed only at the final FFN to preserve KV-cache reuse. The mechanism screen used five methods: shared base, independent rank-2 LoRA per domain, Mirror, Mirror plus rank-1 private residual, and independent full-rank private updates. Each used 1,500 AdamW updates in development/fresh synthetic worlds. Common LR `0.003` was selected on world 25300 only. Fresh worlds 25301–25303 were trained from independent low-rank domain updates. A domain address was explicitly provided to every prediction method, so learned routing/router cost was outside the quality/storage comparison. Actual serialized `torch.save` state-dict plus config bytes were counted.

A separate two-layer causal decoder cache probe compared addresses with a view only after the final layer's attention and with views after both layers' attention. K/V cache differences were measured exactly for an 8-token prefix.

## D — decision

**FAIL for replacing DMoE-style rank-2 private experts with this Mirror coordinate.** In all three fresh worlds, rank-2 LoRA reached MSE `3.6e-14`–`6.1e-14`; Mirror reached `0.0161`–`0.0173`, essentially the shared-base-only result (`0.0163`–`0.0177`). Mirror used 2,956 bytes versus 4,110 for LoRA (28.1% fewer), but failed the predeclared 10% quality gate by a wide margin.

Adding per-domain rank-1 private residuals reduced MSE to `0.00369`–`0.00469` (about 71–78% below base-only and 71–77% below Mirror-only), at 3,919 bytes. That remains far above the rank-2 LoRA result. Independent full-rank residuals reached `5.7e-7`–`3.8e-6` using 6,993 bytes. This is evidence that some private capacity is needed for these independent domain updates; rank 1 was insufficient to match their rank-2 structure.

**Cache placement gate PASS.** Final-only Mirror changed no cached K or V value across the two domain addresses (maximum absolute difference 0). With an early-layer view, layer-0 K/V still matched, while layer-1 K/V differed by `0.3494`. The placement mechanism behaves as predicted in the tiny causal decoder.

Compute/runtime regressed in this implementation: Mirror training took `9.6–11.9 s` versus `1.3 s` for rank-2 LoRA at equal updates; single-thread CPU inference throughput was variable across worlds (Mirror/LoRA ratios `0.49–1.06`). The analytical MAC proxy was close, indicating Python coordinate-op overhead is not captured by MAC count. This is not an optimized runtime claim.

All 15 fresh result rows replayed with maximum absolute MSE difference `4.4e-12`; serialized bytes matched. Tests: 4 passed. See `CACHE_AUDIT.json`, `RESULTS_CORE.csv`, and `VERIFICATION.json`.

## C — strongest counter-hypothesis

The task teacher is specifically generated from independent rank-2 LoRA updates, so it directly tests a case where per-domain private low-rank weights should work and a conjugate shared-matrix view need not. The result bounds this Givens family on independent updates; it does not rule out other Mirror coordinates or domains whose functions share a compact transform structure.

## U — unconfirmed

- Whether natural-language knowledge experts have shared coordinate structure that survives this comparison.
- Whether a learned uncertainty router changes the quality/storage frontier.
- Whether a fused view kernel removes the measured CPU overhead.
- External expert loading/offloading latency, real tokenizer/model cache behavior, and GPU runtime.

## Fact / interpretation / hypothesis

- **Fact:** the synthetic quality, bytes, update counts, wall times, MAC proxy, and exact K/V differences above were measured and replayed under the frozen protocol.
- **Interpretation:** final-layer placement preserves prefix KV caches in this architecture, while the tested view does not encode independently generated domain LoRA updates efficiently enough to replace private experts.
- **Hypothesis:** Views may be useful where domain updates lie near a low-description coordinate orbit; that structure was deliberately absent from these independent rank-2 teacher updates and remains untested in natural data.
