# T001 — SmolLM2-135M Real-Model Pilot

This directory is the first real-model FQC pilot. It is currently **preflight
only**: no checkpoint tensor has yet been loaded by the canonical extractor in
this repository's recorded evidence.

## Pinned source

- model: `HuggingFaceTB/SmolLM2-135M`
- immutable Hub snapshot: `28e66ca6931668447a3bac213f23d990ad3b0e2b`
- checkpoint: `model.safetensors`
- checkpoint SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- checkpoint size: 269,060,552 bytes

See `pilot_pin.json` for the machine-readable configuration and preflight bit
budget.

## Preflight accounting

The supported Llama configuration implies 134,515,008 unique scalar parameters
when the tied embedding/lm-head storage is counted once. This is a cross-check,
not the authoritative denominator. The real denominator will be emitted only by
the live storage inventory.

Under the project's 16-bit baseline convention, the config-derived hard 64x
budget is 33,628,752 bits = 4,203,594 bytes.

The published safetensors file is 30,536 bytes larger than the raw 16-bit scalar
payload implied by this count, which is consistent with a small container/header
metadata overhead. This observation is not a codec result.

## Next gate

1. Load the pinned checkpoint in an environment with model-file access.
2. Verify the downloaded file SHA-256.
3. Build the adapter plan and live D57 storage manifest.
4. Require live `N == 134,515,008`; mismatch is a hard investigation trigger.
5. Repeat extraction in a fresh process and compare public manifest hashes.
6. Run layer-0 module replay under a predeclared tolerance contract.
7. Only then run D58/D61 structural and functional diagnostics.

## Current environment note

During reconstruction on 2026-09-02, the available Hugging Face remote-compute
endpoint returned HTTP 402 and the local execution container had no external DNS.
Therefore no weight-derived result is recorded here. The source pin and hash are
fixed so the exact same pilot can resume in a weight-accessible environment.
