# MA-574 results and audit

## Fact

The committed source has SHA-256 `7e3e95138fc981879086bf20abfefb3c2b1061882a981b703329300df4ea032f`, while `freeze.json` records `c88459769088f84ce787dd38c593f827c805bbc2285032c84f4c74450dad5e3b`. The committed protocol digest matches its recorded freeze value. Same-seed diagnostic replays select codebook candidate IDs `[5,0,1,2]` for 57401 and `[3,0,1,2]` for 57402; the files labeled registered development report `[0,1,2,3]` for both. Replay outputs preserve the native shared-codebook alias and have near-identical aggregate matrix reconstruction metrics. Actual serialized artifact sizes and hashes are recorded in `ARTIFACT_PROVENANCE.json`; replay files are held outside Git under `/workspace/artifacts/ma574_replay_57401` and `...57402`.

## Interpretation

The preregistered evidence chain is not auditable end to end. The available development artifacts suggest shared, random, and native codebooks produce identical serialized payloads, but the digest and same-seed selection mismatch makes that registered comparison unreliable. The diagnostic replay supports the same native-control concern but cannot replace frozen registered evidence.

## Hypothesis

A four-entry rotation codebook may save code metadata relative to independent per-matrix rotations, but this experiment does not establish a Mirror-specific quality/storage gain.

## H / T / D / C / U

**H:** a shared learned rotation codebook transfers quantization quality to held-out layers while lowering actual total payload bytes; native codebook selection is the key attribution control.

**T:** pinned Pythia-70M matrices, 12 attention-dense/MLP-up weights from layers 0–5, groupwise int4, 16 candidate rotations, four code entries, layer split 0–3/4–5, and seven quantization controls. Registered development artifacts and committed-source same-seed diagnostic replays were audited. No fresh seed was opened.

**D:** NOT ESTABLISHED due to the frozen source digest mismatch and different codebook IDs on same-seed replay. The available artifact set suggests an exact native/random alias but is not sufficient for a registered verdict.

**C:** the observed metadata savings and quantization result may be explained entirely by ordinary codebook selection/QuaRot-style gauge choice.

**U:** reproducible registered dev outputs from a verified freeze; language-model NLL/perplexity, optimized SpinQuant, end-to-end inference latency, and fresh-seed behavior.
