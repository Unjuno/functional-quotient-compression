# MA-877 — position-free KV transfer after inverse RoPE

Status: SCREENING  
Evidence lane: cross-model KV / quality / storage / compute  
Base: `66aa8caa`

## Hypothesis

For Pythia-70M→SmolLM2-135M, inverse-RoPE source and target keys before per-role affine ridge fitting, then reapply target RoPE, should improve transfer at positions beyond the calibration window. A rank-32 shared map basis should retain quality within 0.05 nat while reducing actual mapper bytes by at least 50%.

## Insertion and controls

The Mirror coordinate is the per-layer/head coefficient vector used to reconstruct one K/V affine map from a shared map basis. Controls are native target prefill, independent position-free affine maps, direct RoPE-space affine maps, and PCA map coding. Since PCA is the native shared-basis control, any exact match is not Mirror-specific.

## Protocol

See [PROTOCOL.json](PROTOCOL.json). Development uses Wikitext train only. Fresh validation calibration and test prompts are disjoint and fixed before fresh evaluation. Calibration maps see only positions 0–63. Three fresh seeds are 983, 997, and 1009. CPU-only.

## Status

The exact model-specific inverse/reapply algebra is tested against both cached model implementations. Full development and fresh measurements are recorded after the frozen settings are committed.
