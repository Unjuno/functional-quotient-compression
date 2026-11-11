# MA-877 — position-free KV transfer after inverse RoPE

Status: FAIL on frozen quality/storage frontier; replay reproduces all gate verdicts.
Evidence lane: cross-model KV / quality / storage / compute  
Branch: research/ma-877-position-free-kv-transfer-20261111  
Base: 66aa8caa

## Hypothesis

For Pythia-70M→SmolLM2-135M, inverse-RoPE source and target keys before per-role affine ridge fitting, then reapply target RoPE, should improve transfer at positions beyond the calibration window. A rank-32 shared map basis should retain quality within 0.05 nat while reducing actual mapper bytes by at least 50%.

## Insertion and controls

The Mirror coordinate is the per-layer/head coefficient vector used to reconstruct one K/V affine map from a shared map basis. Controls are native target prefill, independent position-free affine maps, direct RoPE-space affine maps, and PCA map coding. Since PCA is the native shared-basis control, any exact match is not Mirror-specific.

## Results

Inverse RoPE improved full-map NLL over direct RoPE-space ridge in each fresh seed, but full maps remained +0.82/+2.97/+0.87 nat above native prefill. Rank32 remained +12.35/+16.44/+13.61 nat above native. Its actual serialized payload was 573,125 B vs 2,996,165 B independent, an 80.871% reduction. The frozen quality gates failed.

See [RESULTS.md](RESULTS.md), [RESULTS_CORE.csv](RESULTS_CORE.csv), and [fresh_summary.json](fresh_summary.json).

## Verification boundary

Four tests pass, including payload serialization size/determinism. Model-cache RoPE roundtrips pass at <9e-16. A fresh replay reproduces prompt IDs, exact payload bytes, and all gate verdicts; NLL values vary by at most 0.02051181 nat across CPU runs. See [VERIFICATION.json](VERIFICATION.json).

Development used Wikitext train only. Fresh validation calibration and test prompts were disjoint and frozen before evaluation. Calibration maps saw positions 0–63. Seeds were 983, 997, and 1009. CPU-only; no GPU claim.
