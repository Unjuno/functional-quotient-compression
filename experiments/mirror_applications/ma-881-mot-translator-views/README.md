# MA-881 — MoT translator experts with shared Views

Status: FAIL. Prior art: PA238 (Mixture-of-Translators), with PA237 weighted-ridge mapper controls.

Two token-routed cross-model KV translator experts were compressed to a shared rank-32 map basis and per-expert coordinates. The payload shrank by 89.98%, but heldout NLL quality failed in all three fresh worlds. See [RESULTS.md](RESULTS.md).

## Method boundary

The screen retains token-level Top-1 routing and target-side replay KV correction but uses per-role affine ridge maps as translators. It is a component-level adaptation, not a full MoT architecture reproduction. The shared-basis view codes are ordinary PCA coefficient coding; no Mirror-specific benefit is established. Details are frozen in [PROTOCOL.json](PROTOCOL.json).
