# T001 — SmolLM2-135M Real-Model Pilot

This directory is the first real-model FQC pilot. The **serialized checkpoint
header gate has passed on the actual pinned `model.safetensors` artifact**.
Runtime model replay and structural compression measurements have not yet been
performed.

## Pinned source

- model: `HuggingFaceTB/SmolLM2-135M`
- immutable Hub snapshot: `28e66ca6931668447a3bac213f23d990ad3b0e2b`
- checkpoint: `model.safetensors`
- checkpoint SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- checkpoint size: 269,060,552 bytes

See `pilot_pin.json` for the pinned source configuration and
`results/header_preflight_result.json` for the recorded actual-artifact result.

## Actual serialized-checkpoint result

GitHub Actions downloaded the immutable checkpoint and verified:

- status: **PASS**
- serialized tensors: **272**
- serialized scalars: **134,515,008**
- dtype inventory: **134,515,008 BF16 scalars**
- missing checkpoint keys: **0**
- unaccounted checkpoint keys: **0**
- safetensors header: **30,528 bytes**
- 8-byte prefix + header: **30,536 bytes**
- tensor payload: **269,030,016 bytes**
- total file: **269,060,552 bytes**

The entire difference between the BF16 tensor payload and the file size is
therefore exactly the safetensors 8-byte length prefix plus JSON header for this
artifact.

## Baseline denominator and 64x target

For the pinned source artifact, the serialized tensor inventory is the authority
for the baseline paid scalar payload. It establishes

`N = 134,515,008`

rather than merely predicting it from config metadata. Under the project 16-bit
baseline convention:

- baseline payload: 2,152,240,128 bits = 269,030,016 bytes
- hard 64x budget: **33,628,752 bits = 4,203,594 bytes**

Runtime storage inventory remains required as a consistency check for tied roles
and loader semantics, but loader-specific duplication must not inflate the source
baseline denominator.

## Evidence boundary

This PASS proves serialized checkpoint identity, tensor coverage, shapes/dtypes
as checked by the adapter, and the source-artifact scalar denominator. It does
**not** prove:

- runtime storage aliasing,
- module replay correctness,
- shared low-description structure,
- functional sensitivity,
- any FQC compression ratio,
- or 64x feasibility.

## Next gate

1. Load the pinned checkpoint in a weight-accessible runtime.
2. Build the runtime D57 storage inventory and check consistency with the source
   artifact without allowing loader duplication to redefine the denominator.
3. Repeat extraction in a fresh process and compare public manifest hashes.
4. Run layer-0 module replay under a predeclared tolerance contract.
5. Only after replay PASS, run D58/D61 structural and functional diagnostics.
6. Generate the first shared-root/private-residual codec candidates.

## Execution provenance

The successful header preflight ran in GitHub Actions because the local research
container lacked external DNS and the available Hugging Face remote-compute
endpoint returned HTTP 402. This environment limitation no longer blocks
serialized-checkpoint evidence; live model execution remains a later gate.
