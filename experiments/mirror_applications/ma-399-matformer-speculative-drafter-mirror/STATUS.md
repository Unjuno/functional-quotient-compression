# MA-399 status

- Status: **FAIL** at frozen acceptance, training-cost and expected-throughput gates; fresh remains sealed.
- Branch: `research/ma-399-matformer-speculative-drafter-mirror-20261009`
- Base commit: `f7cabacf`
- Prior art: PA53 MatFormer; PA15 routed verifier.
- Development seeds: 39901, 39902.
- Fresh seeds: 39911, 39912, 39913 (sealed).
- Serialized payloads: twelve; sizes, SHA-256, acceptance and exactness metrics replay.
- Tests: four passed.

## Result

Mirror overlap improved width-8 by only .0016–.0028 and was below FiLM/width-12; exact residual-corrected distributions remained within 1.5e-7. Expected Mirror throughput was .642/.652× full-only, versus nested width-8 at 1.042/1.056×. Mirror payload was below independent pair bytes, but above shared nested-only. Fresh stayed sealed. See README and RESULTS_CORE.csv.
