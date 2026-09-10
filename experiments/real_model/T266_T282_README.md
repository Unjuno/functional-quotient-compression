# T266-T282 real-model experiment index

This directory-level index records the canonical status of the real-checkpoint phase without dumping large handoff packages into git.

## Experiment groups

### T266-T272 — whole-model serialization and 64x failure boundary

Purpose:
- connect supplied TinyStories checkpoints to an actual serializer/decoder path;
- produce a hard-size 64x artifact;
- evaluate selective private correction;
- validate integrity failures.

Canonical conclusion:
- engineering codec path: **PASS**;
- quality-preserving 64x: **FAIL for the tested simple family / not established generally**;
- selective support matters; larger support is not monotonically better.

### T273-T278 — stronger non-sharing affine baseline

Purpose:
- strengthen the quantization control with activation-sensitive affine fitting;
- compare parameter-space and functional objectives;
- reproduce all generated artifacts from checkpoint inputs;
- harden schema/integrity checks.

Canonical conclusion:
- activation-aware fitting improves the internal non-sharing baseline on the tested checkpoints;
- lower parameter reconstruction error is not a sufficient proxy for task quality;
- engineering reproduction: **PASS**;
- NLL noninferiority/generalization: **UNCERTAIN**.

### T279-T282 — frozen-candidate pilot and implementation audit

Purpose:
- freeze the T278 candidate before a later public-prompt-derived pilot;
- verify long-context/local-attention/cache behavior;
- compare final file sizes after the same reversible outer compression;
- replay fixed evaluations in separate processes.

Canonical conclusion:
- the storage-oriented frozen candidate retained a smaller-file/lower-KL direction on the pilot;
- the pilot is not the official TinyStories validation corpus and is not sufficient for a publication-quality quality claim;
- final serialized bytes can reorder equal-width logical codecs;
- engineering replay/integrity coverage reached 153 tests.

## Required external inputs

Large model checkpoints are intentionally excluded. Reproduction requires the supplied TinyStories 1M/3M/8M/28M checkpoint archives or equivalent verified upstream inputs. Implementations must verify source hashes before treating results as reproduction.

## Canonical outputs kept in git

Keep:
- compact machine-readable claim/evidence summaries;
- source and environment provenance;
- small scripts/tests after normalization into the repository API;
- aggregate and per-example metrics when licensing permits;
- hashes for large external artifacts.

Do not keep:
- duplicated checkpoints;
- large handoff ZIPs;
- large generated binary envelopes solely for convenience;
- transient logs/process IDs;
- claims that silently upgrade pilot data to official validation.

## Next experiment contract

Do not tune a new FQC sharing method against the final audit split.

At matched **final serialized bytes**, evaluate:

1. strong activation-aware non-sharing control;
2. control + functional sharing;
3. sharing + selected private exceptions;
4. private exceptions + joint allocation / QCO.

Measure token-weighted NLL and KL over a multi-rate frontier. Freeze the candidate family before the untouched final audit. Record failures as first-class results.

See:
- `docs/REAL_MODEL_T266_T282.md`
- `docs/RESEARCH_STATE.md`
- `claims/T266_T282_REAL_MODEL_EVIDENCE.json`
