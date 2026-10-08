# MA-879 — Attention-sensitive calibration of Mirror KV translators

Status: SCREENING

Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA237 (CacheBridge), PA227 (Fisher-geometric traces)

## H — falsifiable hypothesis

With identical shared KV bases and code bytes, attention-sensitive calibration will preserve causal attention outputs better than raw cache MSE for task-conditioned Mirror translators. The effect should appear on fresh aligned contexts and be reported against independent attention-calibrated mappers and ordinary basis mixing.

## T — frozen scope

Synthetic causal attention with four contexts, eight tokens and head dimension four. Compare independent raw-MSE and attention-calibrated mappers, Mirror codes trained with each loss, a two-basis control, and hard sharing. The aligned teacher uses context-specific Givens views; an unrelated-map teacher marks the boundary. Draw32 replay is in `source/draw32_exclusions.json`; full settings/seeds/gates are in `PROTOCOL.json`.

## D — decision

Pending development and fresh audit.

## C — strongest counter-hypothesis

Attention weighting may only help when the target mapping lies on the assumed Mirror orbit; ordinary attention-calibrated translation may work just as well or better with fewer restrictions.

## U — boundaries

Synthetic mechanism only: no pretrained models, autoregressive NLL/perplexity, GPU fused attention, natural cache transfer, or production storage/runtime claim.
