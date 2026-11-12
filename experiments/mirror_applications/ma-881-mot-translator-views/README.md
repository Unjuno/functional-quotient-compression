# MA-881 — MoT translator experts with shared Mirror Views

Status: SCREENING. Prior art: PA238 (Mixture-of-Translators), with PA237 weighted-ridge mapper controls.

## Hypothesis

A shared map basis and small expert coordinates can compress two token-routed translators while preserving MoT-style target context correction and heldout target NLL.

## Method boundary

The screen retains PA238's token-level Top-1 router and target-side replay KV correction loss, but uses per-role affine ridge maps as the two expert translators. It is a component-level adaptation for small cached LMs, not a full MoT architecture reproduction. The shared-basis view codes are the ordinary PCA/low-rank control; no Mirror-specific claim follows from a pass alone.

All pair, prompt, router, correction-loss, serialization and gate details are frozen in [PROTOCOL.json](PROTOCOL.json).
