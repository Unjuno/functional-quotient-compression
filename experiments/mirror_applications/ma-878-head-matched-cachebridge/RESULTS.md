# MA-878 results — FAIL

## H — hypothesis

On the frozen Pythia-70M→SmolLM2-135M pair, choosing a one-to-one source KV-head support by target-attention-weighted calibration loss will improve fresh target next-token NLL over fixed source heads [0,4,7] by at least 0.10 nat in each of three fresh partitions. A rank-32 shared PCA map basis will remain within 0.05 nat of matched independent maps and within 0.50 nat of native target while using at most half their actual serialized translator bytes.

## T — execution

**Fact.** Protocol commit `e6a164427071f3b424d823e727e4a1f96eb6e9c3`; fresh rank/splits frozen in `e9b36b275` before fresh access. Pythia-70M (6 layers, 8 KV heads)→SmolLM2-135M (30 layers, 3 KV heads); Wikitext-2 raw. Attention-weighted affine ridge alpha 0.01, 64 calibration positions, depth-nearest layer pairing, source/target keys inverse-RoPE'd during fitting and target RoPE reapplied. Development seed 829 used 24 train calibration / 16 train evaluation prompts. Rank 32 was selected by the frozen minimum-development-NLL rule among ranks 1/2/4/8/16/32.

**Fact.** Fresh partitions used validation calibration/test evaluation with seeds 983, 997, 1009; each had 12 calibration and 16 test examples. No prompt row IDs overlap across these partitions or between calibration/test. Model snapshots: Pythia `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (166,029,852 weight bytes); SmolLM2 `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (269,060,552 weight bytes). CPU only, CUDA unavailable. Zero optimizer updates. Fresh token counts: calibration 475/616/428; test 587/668/697. Active forward proxy: 129,276 layer-token events total (calibration and prefill positions across 6+30 layers plus three one-token target-cache probes per test example); this omits attention-matrix quadratic weighting and is not FLOPs. Calibration wall time was 5.49/5.97/4.74 s; PCA basis fit 0.159/0.149/0.112 s. Mean per-example cache application was 16.40/16.40/17.28 ms fixed, 15.31/15.60/19.85 ms matched, and 15.09/16.50/15.74 ms rank-32. These are CPU microbenchmarks, not end-to-end serving throughput.

## D — decision

**Fact.** Fresh mean NLL by seed (native / fixed-spread / matched-full / matched-rank32):

| Seed | Native | Fixed [0,4,7] | Attention-matched full | Matched rank-32 | Matched gain vs fixed | Rank-32 gap vs full |
|---|---:|---:|---:|---:|---:|---:|
| 983 | 3.085 | 5.341 | 5.183 | 17.607 | +0.158 | +12.424 |
| 997 | 1.911 | 4.617 | 5.001 | 12.357 | −0.384 | +7.356 |
| 1009 | 2.850 | 4.471 | 4.688 | 10.328 | −0.217 | +5.640 |
| Mean | 2.615 | 4.810 | 4.957 | 13.431 | — | — |

**Fact.** The actual serialized payload (including header, pair identifier and per-map head/layer/role metadata) was 2,996,165 B for either full-map control and 573,125 B for rank 32, an 80.871% reduction. All three byte checks and all prompt identity checks passed. Three preregistered quality gates fail: matched support improves by ≥0.10 nat in only one of three seeds; rank 32 is 5.64–12.42 nat worse than full maps; rank 32 is 7.48–14.52 nat worse than native. Verdict: **FAIL**.

## C — strongest counter-hypothesis

The source and target models have substantially different representations. Matching head support can sometimes help, but an affine translator fitted on a small calibration set does not recover target cache geometry; the shared PCA basis further destroys task-relevant directions. The large rank-32 quality collapse is consistent with poor low-rank approximation, while the full-map native gap indicates that head selection alone cannot solve cross-model cache incompatibility. This is a scoped interpretation, not a general impossibility result.

## U — remaining uncertainty

One ordered model pair and one corpus do not establish whether this failure is specific to the architectures, calibration size, or source/target scale mismatch. The matched-head assignment is selected on calibration residuals; a larger calibration budget or a nonlinear bridge could alter quality but would be a new experiment. No GPU latency, generation-trajectory quality, or paper-faithful CacheBridge reproduction was measured.

## Fact / interpretation / hypothesis

- **Fact:** exact metrics, payloads, snapshots, prompt rows and runtime observations are recorded above and in `fresh_results.json` / `dev_results.json`.
- **Interpretation:** head matching does not reliably improve over fixed spread, and rank-32 PCA is not a useful bridge at this byte point on this pair.
- **Hypothesis:** cross-model KV/head transfer needs either a much richer/private mapper or a more representation-compatible model pair; a compact shared map basis alone is inadequate here.
