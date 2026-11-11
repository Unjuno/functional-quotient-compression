# MA-878 — attention-weighted head-matched KV bridge

Status: SCREENING  
Branch: `research/ma-878-head-matched-cachebridge-20261111`  
Base commit: `e705db936`

## Hypothesis

For Pythia-70M→SmolLM2-135M, an injective source KV-head assignment selected with target-attention-weighted calibration should beat fixed spread head support on fresh target NLL. A rank32 shared map basis should retain that quality while reducing actual serialized mapper bytes by at least 50%.

## Controls

1. Native target prefill.
2. Fixed spread source heads [0,4,7] with attention-weighted affine ridge maps.
3. CacheBridge-style calibration-selected one-to-one head support with the same weighted ridge fitting.
4. Rank sweep over the matched maps, with native PCA shared-map coding as the attribution control.

Both support methods use the same inverse-RoPE/content-space map interface established in MA-877; only the source-head support changes. Target-token importance is the sum of attention received over causal queries and grouped query heads, normalized to mean one per prompt.

## Protocol

See [PROTOCOL.json](PROTOCOL.json). Development uses Wikitext-2 train, seed 829. Fresh validation calibration and test prompts are seed-disjoint and fixed before evaluation (983, 997, 1009). Calibration fitting is limited to positions 0–63. All learned maps and head indices are charged in actual serialized payload bytes.

This is a narrow operational screen motivated by PA237, not a reproduction of CacheBridge.
