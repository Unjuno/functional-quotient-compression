# MA-880 results — FAIL

## H — hypothesis

A rank-32 canonical KV-map basis plus a low-rank source-model × target-model × normalized layer/head/KV CP coordinate would predict useful maps for whole ordered pairs unseen during factor fitting. Each factor prediction had to stay within +0.50 nat of an independent full ridge map and +1.00 nat of target-native prefill, while using at least 50% fewer actual serialized bridge bytes.

## T — treatment and controls

Four cached pretrained models were tested: GPT-2, Pythia-70M, Pythia-160M and SmolLM2-135M. The training cycle had four ordered pairs; two other whole pairs were used only for development rank selection; the two reverse-cycle fresh pairs were Pythia-160M→Pythia-70M and GPT-2→SmolLM2. All model IDs appeared as both source and target in training pairs. The CP rank selected on development was Q=1, with 600 Adam updates at LR .01. The rank-32 basis and per-code standardization used only training-pair map codes.

Fresh validation calibration used 12 prompts per training pair and 12 prompts per heldout-pair full-ridge control; test used 12 prompts per heldout pair in seeds 983, 997 and 1009. Source/target tokenizer character-offset sequences had to match. Controls were native target prefill, independent affine ridge maps per fresh pair, and rank-32 PCA coding with heldout-pair calibration maps. CPU only; no GPU serving claim.

Code review after the first fresh run found that the initial implementation omitted the protocol-declared RoPE removal/reapplication. That first run, its settings, results and payloads are preserved as `fresh_results_unrotated_deviation.json` and `FROZEN_SETTINGS_UNROTATED.json`; they did not select corrected settings. Amendment 03 restored the already specified RoPE rule, reran rank selection on train-only development pairs, froze Q=1, and then reran the same fresh prompt IDs. Because the corrected rerun reuses the same validation/test rows as the implementation-deviation run, it is an amended diagnostic, not an independent second audit. The unrotated output is excluded from the official gate decision.

## D — decision: FAIL

| Seed | Fresh pair | Native NLL | Full ridge | Factor CP | PCA code | Factor − full | Factor − native | Factor payload / full payload | Saved |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 983 | pythia-160m-deduped → pythia-70m-deduped | 3.237 | 4.194 | 42.845 | 23.819 | 38.65 | 39.61 | 583,915 / 4,594,139 B | 87.29% |
| 983 | gpt2 → SmolLM2-135M | 2.383 | 6.342 | 16.079 | 16.731 | 9.74 | 13.70 | 583,915 / 4,594,139 B | 87.29% |
| 997 | pythia-160m-deduped → pythia-70m-deduped | 3.006 | 5.147 | 39.393 | 19.383 | 34.25 | 36.39 | 583,915 / 4,594,139 B | 87.29% |
| 997 | gpt2 → SmolLM2-135M | 3.003 | 5.708 | 20.831 | 14.528 | 15.12 | 17.83 | 583,915 / 4,594,139 B | 87.29% |
| 1009 | pythia-160m-deduped → pythia-70m-deduped | 3.129 | 4.986 | 47.182 | 20.563 | 42.20 | 44.05 | 583,915 / 4,594,139 B | 87.29% |
| 1009 | gpt2 → SmolLM2-135M | 1.977 | 5.455 | 19.787 | 15.581 | 14.33 | 17.81 | 583,915 / 4,594,139 B | 87.29% |

Across the six heldout pair×seed conditions, mean NLL was native 2.789, full ridge 5.305, PCA-code control 18.434, and factor CP 31.019. Factor CP was 9.74–42.20 nat worse than full ridge and 13.70–44.05 nat worse than native in every condition. It was worse than PCA in five of six conditions; in one it improved by 0.65 nat. Factor-minus-PCA ranged from -0.65 to +26.62 nat (mean +12.59). Both preregistered quality gates fail in 6/6 cases. Full independent maps themselves were 0.96–3.96 nat above native; only one of six was within +1.00 nat.

The byte gate passed in all three worlds: the complete factor payload for both logical pairs was 583,915 B versus 4,594,139 B for independent maps (87.29% fewer bytes; 7.87× smaller). The payload charges model and pair IDs, explicit layer/head/KV references, role-grid dimensions, shared mean and basis, code normalization means/scales, source/target factors, role factors, bias and format header. Model weights (1,358,194,271 B total) are shared and reported separately. Factor bytes do not include the independent per-pair map controls.

### Fact / Interpretation / Hypothesis

- **Fact:** factor CP reaches mean heldout NLL 31.019, compared with 5.305 for independent full ridge and 2.789 for native target prefill. It is worse than the pair-calibrated PCA control by 12.59 nat. Payload falls 87.29%; factorized KV application itself did not improve quality or beat the PCA control.
- **Interpretation:** source×target CP coordinates compress translator state but do not recover useful unseen-pair functions in this four-model screen. The storage point is far outside the quality frontier. The CP decoder is conventional tensor factorization, so this gives no Mirror-specific advantage. Direct full-map application is itself unreliable relative to native prefill in most heldout conditions.
- **Hypothesis:** four directed training edges do not identify the pair-specific, role-varying cache maps needed for reverse directions. A shared rank-32 basis loses relevant map structure, and rank-1 interaction cannot express enough pair×role variation. More pair coverage, private residuals or nonlinear translation may help but add state and were not tested.

## C — strongest counter-hypothesis

The factorization is underdetermined by only four training directions, and its shared role grid may collapse layer/head distinctions. The failure does not establish that pair-factorization cannot work with denser model-pair coverage, a larger interaction rank, private residuals or MoT-style target correction. The full ridge quality itself varies by pair, so some loss is a translation-family limitation rather than compression alone.

## U — unknowns

Only four models, two fresh ordered directions, Wikitext-2 next-token probes, one fixed head/depth rule, head dimension 64, rank 32 and a single development-selected Q were tested. Multi-token trajectory quality, longer contexts, broader model families, denser training-pair graphs, Q>1 fresh quality, nonlinear/private repair, confidence fallback, GPU latency and factor-map decode cost are untested. The fresh split was replayed after a predeclared implementation correction and is not an independent audit of corrected code.

## Compute and artifacts

Fresh world wall times were 33.9s, 33.5s, 36.1s (mean 34.5s). Across worlds, training-pair ridge fitting took 11.9s, fresh full-map control fitting took 6.3s, and CP fitting took 1.06s. Peak RSS was 2848 MiB; CUDA unavailable. Precomputed-map application time averaged 13.0 ms per prompt; factor-map decoding time is not included. Exact factor payload bytes and hashes are retained under `artifacts/`; model snapshot hashes/byte sizes and prompt IDs are in `fresh_results.json`.
