# MA-879 — Attention-sensitive calibration of Mirror KV translators

Status: SCREENING

Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA237 (CacheBridge), PA227 (Fisher-geometric traces)

## H — falsifiable hypothesis

With identical shared KV bases and code bytes, attention-sensitive calibration will preserve causal attention outputs better than raw cache MSE for task-conditioned Mirror translators. The effect should appear on fresh aligned contexts and be reported against independent attention-calibrated mappers and ordinary basis mixing.

## T — frozen scope

Synthetic causal attention with four contexts, eight tokens and head dimension four. Compare independent raw-MSE and attention-calibrated mappers, Mirror codes trained with each loss, a two-basis control, and hard sharing. The aligned teacher uses context-specific Givens views; an unrelated-map teacher marks the boundary. Draw32 replay is in `source/draw32_exclusions.json`; full settings/seeds/gates are in `PROTOCOL.json`.

## D — decision

**PASS on the frozen aligned attention-calibration gate; operational status PROMISING.** On 3 fresh aligned worlds × 3 model seeds × 4 contexts, Mirror attention calibration lowered output MSE versus the same Mirror code trained on raw KV MSE in all 3 worlds: `0.0` (numerically near zero) vs `1.33e-6` pooled. Both versions serialize to 336 B. Attention KL and target argmax-key NLL were no worse than the independent attention-trained mapper; Mirror NLL was 0.74139 vs 0.74277. The CacheBridge-style weighted ridge control also achieved near-zero functional error at 688 B, while the rank-2 basis was near exact at 512 B. Thus the Mirror translator is a smaller storage point, not a quality win over these controls.

### Fact

On the independent-map boundary, Mirror attention output MSE was 0.3187 and target-key NLL 1.0935, while weighted ridge was near zero and NLL 0.7588. The Mirror payload was 336 B versus 688 B for independent mappers. CPU batch-1 latency was 0.404 ms for Mirror, versus 0.0087 ms weighted ridge and 0.030 ms rank-2 basis. The current eager implementation builds all four view matrices before selecting a context.

### Interpretation

Attention-weighted calibration reduces a small but repeatable functional error at equal Mirror bytes in this aligned synthetic setting. Raw cache MSE was already nearly exact, so absolute gains are tiny. Weighted ridge obtains exact translation because the teacher is linear, and is much faster; rank-2 basis also nearly recovers the aligned orbit with a larger payload than Mirror.

### Hypothesis

This supports attention-sensitive calibration as a useful objective for compact Mirror codes when task maps lie on the assumed Givens orbit. It does not establish natural cross-model cache transfer or language-model NLL/perplexity improvements. Independent maps show the private-capacity boundary.

All 432 task-level rows and actual payload hashes are retained. A second audit run exactly reproduced cache/output MSE, KL, NLL and payload bytes.

## C — strongest counter-hypothesis

The teacher is explicitly on the Mirror orbit, and raw-cache fitting already gives nearly exact attention behavior. CacheBridge-style weighted ridge is exact, roughly 46× faster on CPU, and only costs 352 B more; the mechanism may not generalize beyond the aligned synthetic setup.

## U — boundaries

Synthetic mechanism only: no pretrained models, autoregressive NLL/perplexity, GPU fused attention, natural cache transfer, or production storage/runtime claim.
