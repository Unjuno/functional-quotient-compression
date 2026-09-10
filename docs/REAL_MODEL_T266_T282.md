# Real-model evidence: T266-T282

This document is the canonical compact record of the first end-to-end real-checkpoint codec phase. Large checkpoints, handoff ZIPs and duplicated binary artifacts are intentionally excluded from git history.

## Scope

Models: supplied TinyStories 1M / 3M / 8M / 28M checkpoints.

Primary engineering target: serialize learned Transformer weights into an actual codec artifact, independently decode them, run the reconstructed model, and measure quality and actual stored bytes under explicit provenance.

Primary scientific target: determine whether task-aware / functional allocation and later FQC-specific sharing can improve rate-distortion beyond strong non-sharing controls.

## T266-T272 — first whole-model serialized codec

Established:

- actual 28M whole-model `.fqc` artifacts were produced;
- 108 learned tensors were independently decoded and used for whole-model forward evaluation;
- a hard 64x-size artifact was physically constructed;
- integrity checks rejected truncation, trailing data, payload corruption and inconsistent headers.

Key negative result:

- the tested simple 64x family failed quality badly and is **not** evidence of quality-preserving 64x compression.

Key conditional result:

- a selective 960-neuron private correction improved the failed 64x candidate relative to no correction under the same target size;
- expanding support to 2,363 neurons made the tested candidate worse, so "spend every remaining bit on more corrections" is rejected as a general policy.

Metric warning:

- KL and token NLL can move in different directions. Candidate selection must therefore not treat KL-only improvement as task-quality proof.

## T273-T278 — stronger non-sharing affine baseline and reproduction

A stronger scalar/affine non-sharing control was built using activation-sensitive fitting. This line is a control improvement, **not FQC novelty**.

On the tested checkpoints, activation-aware fitting improved the internal non-sharing baseline. A 1M counterexample also showed that improving parameter-space reconstruction error can worsen task KL/NLL.

The 28M storage-oriented candidate reduced raw serialized size relative to the group-64 control while improving KL on the retained regression probes. NLL evidence was mixed and was not promoted to a noninferiority claim.

Reproducibility work:

- 27 artifacts were regenerated from checkpoint inputs in a fresh output tree;
- hashes, decoded tensor hashes and per-document metrics matched the retained references;
- 97 tests passed at T278;
- checksum-valid but semantically inconsistent model metadata was added as a negative control.

## T279-T282 — frozen-candidate external-input pilot and implementation checks

The T278 candidates were frozen before evaluation on a later public-prompt-derived pilot set. The pilot was not used to refit or reselect the candidates.

For the 28M storage-oriented frozen candidate, the later pilot preserved a smaller-final-file / lower-KL direction relative to the corresponding non-sharing control after the same outer XZ compression. Mean NLL also improved on that pilot, but this is **preliminary evidence only** because the set is small, correlated and not the official TinyStories validation corpus.

Additional engineering checks:

- long-context attention boundaries were exercised up to model context limits;
- stored attention masks, independent reference calculations and SDPA-compatible calculations were compared;
- KV-cache chunked execution was compared with one-shot execution;
- causal negative controls were exercised;
- reversible outer compression was round-trip checked;
- total regression/integrity coverage reached 153 tests by T282.

## Claim boundary

### Supported

1. A real learned-weight Transformer codec artifact can be serialized, independently decoded and executed end-to-end.
2. Final serialized bytes, not logical code width, can change codec ordering.
3. Parameter-space reconstruction error is not a sufficient task-quality objective.
4. Private correction support is a selection variable; more support can hurt.
5. A stronger activation-aware non-sharing control is necessary before attributing gains to FQC.

### Not supported yet

1. Quality-preserving 64x real-Transformer compression.
2. FQC-specific functional sharing beating a strong non-sharing baseline.
3. Official TinyStories validation superiority.
4. Superiority over strong published methods such as GPTQ/AWQ/AQLM-family baselines.
5. MPS/CUDA runtime, VRAM or energy advantage.
6. A scale law across 1M/3M/8M/28M; each scale currently lacks independent training replicas.

## Next preregistered comparison

At matched final serialized bytes, compare:

- **A — Control:** strong activation-aware non-sharing codec;
- **B — Sharing:** A + FQC functional sharing;
- **C — Private:** B + selected private exceptions;
- **D — Joint:** C + joint byte allocation / QCO.

Primary quality measures should include token-weighted NLL and KL. Candidate selection data and final audit data must remain separate. A multi-rate frontier should be measured rather than optimizing only the 64x point.

## Reproduction policy

Canonical git history should contain source, small fixtures, compact result summaries, hashes, environment manifests and tests. Large checkpoints and duplicated binary artifacts remain external inputs/artifacts identified by cryptographic hashes and provenance.
