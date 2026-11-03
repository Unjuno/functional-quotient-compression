# MA-575 results

## Fact

On both registered development seeds, fitted factor IDs collapsed to candidate 0 for all six layers and all 16 input blocks. Held-out NRMSE was 0.0970468802 for independent matrix×block, layer-shared, random factorized, fitted factorized, and native factorized methods. Identity int4 held-out NRMSE was 0.0973779374.

Actual serialized payloads per seed: independent matrix×block 4,439,840 B total / 13,998 B rotation state; fitted factorized 4,428,926 B / 3,084 B; random factorized 4,428,917 B / 3,075 B; native BOFT-style factorized 4,428,926 B / 3,084 B; identity 4,425,842 B / 0 B. The fitted factorization saves 10,914 B (0.246%) total and 77.97% of rotation bytes versus independent. Random factors have the same measured quality and use 9 B fewer than the fitted representation. Fitted/native quantized-weight arrays and selected signs/permutations/IDs are bytewise equal; only a method-label metadata field differs.

Measured candidate error-table generation took 110.94 s and 118.42 s; factor fitting itself took 0.036 s and 0.010 s. Total process wall time was 114.51 s and 122.22 s. Per-method quantize/serialize timing and operation proxy are in the per-seed metric files. Full NPZ sizes and SHA-256 hashes are in `ARTIFACT_PROVENANCE.json`; bulky NPZs remain outside Git under `/workspace/artifacts/ma575_dev_57501_final` and `...57502_final`.

## Interpretation

The preregistered quality and byte gates versus independent matrix×block codes pass in both dev seeds. The experiment FAILS Mirror-specific attribution because the fitted selection exactly aliases the native BOFT-style shared-factor control, and the random factor control matches reconstruction quality while using slightly fewer bytes. The small payload saving is a generic factor-sharing result, not a Mirror-specific gain. The fitted-factor representation is also 3,084 B larger than identity int4 for only a 0.000331 NRMSE improvement.

## Hypothesis

Layer and input-block factors can approximate independent per-matrix quantization gauges with similar held-out reconstruction and less rotation metadata. The stronger claim that a Mirror factor address adds useful value beyond native shared-factor selection is falsified on this matrix screen.

## H / T / D / C / U

**H:** a layer factor shared across two projections, composed with input-block factors shared across layers, can preserve held-out int4 reconstruction quality with materially fewer stored rotation bytes than independent matrix×block codes.

**T:** two registered seeds, pinned Pythia-70M layers 0–5, attention-dense and MLP-up weights, group size 32, int4, 16 signed-permutation candidates around a fixed Hadamard basis. Layers 0–3 were used for calibration and 4–5 for held-out evaluation. Controls were identity, independent, layer-shared, random factorized, fitted factorized, and native BOFT-style factorized. Actual serialized bytes were measured. Fresh 57511–57513 remain sealed.

**D:** FAIL for Mirror-specific attribution. Quality and bytes versus independent storage passed, but exact native parameter/output alias and random-factor parity failed the distinctiveness gate.

**C:** groupwise int4 reconstruction on these Pythia matrices is nearly invariant to the candidate factor choices; ordinary shared-factor state explains the metadata saving.

**U:** natural-text NLL/perplexity, full-model inference latency, larger models, trained continuous SpinQuant/BOFT, and fresh-seed transfer.
