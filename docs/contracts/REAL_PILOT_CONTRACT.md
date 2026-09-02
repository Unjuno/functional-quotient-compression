# Real-Model Pilot Contract

Canonical reconstruction of the D56 real-pilot evidence contract.

## 1. Hard 64× budget

For a declared 16-bit baseline containing `N` **unique paid scalar values**:

`B0 = 16 N` bits.

A serialized candidate achieves at least 64× compression iff:

`B_candidate <= floor(N / 4)` bits.

The integer threshold is authoritative.

## 2. Baseline boundary

`N` is not a framework-reported parameter count. The pilot must explicitly define the baseline serialization boundary:

- included/excluded embeddings and output heads;
- tied/shared storage and whether it is counted once;
- non-trainable floating buffers;
- architecture/config/tokenizer state;
- baseline quantization metadata, if any;
- any external-fixed assets.

The denominator must match what the declared baseline would actually store/transmit.

## 3. Decoder dependency classes

Every object used by decoding or evaluation must be classified as exactly one of:

- `PAID` — serialized in the candidate payload;
- `EXTERNAL_FIXED` — supplied identically to encoder and decoder by the benchmark contract;
- `DERIVED` — reconstructed deterministically from PAID/EXTERNAL_FIXED ancestors in an acyclic decoder DAG.

Unclassified or cyclic dependencies invalidate deterministic bit accounting.

## 4. Analysis primitives are not automatically free

Functional/gauge-reduced operators may be computed during analysis. That does **not** make them zero-bit codec state.

A future codec can charge an analysis primitive at zero transmitted bits only if the decoder derives it from already available ancestors under the frozen protocol.

## 5. Required pilot manifests

A canonical real-model pilot should contain equivalent information to:

### `PILOT_CONTRACT`

- model/checkpoint identity and content hash;
- baseline boundary and unique scalar count `N`;
- `B0`, rational 64× threshold, and integer `floor(N/4)` threshold;
- decoder protocol/version;
- numerical/reproducibility policy;
- quality contract identifier.

### `TENSOR_INVENTORY`

- tensor names, shapes, dtypes;
- unique storage/tie groups;
- baseline inclusion/exclusion;
- structural role.

### `OPERATOR_EXTRACTION_MANIFEST`

- exact analysis primitives;
- formulas and orientation conventions;
- source tensors and architecture assumptions.

### `PAID_ATOM_MANIFEST`

- unique paid atom ID;
- exact/bounded bit length;
- dependencies;
- sharing identity;
- coding/header/alignment policy.

### `QUALITY_CONTRACT`

- evaluation dataset/input identity;
- preprocessing/tokenization;
- metric formula;
- baseline/reference;
- allowed degradation or required target;
- deterministic/statistical witness semantics;
- randomness and numerical precision policy.

### `REPLAY / AUDIT CONTRACT`

When replay/audit is used, bind workload identity, artifact hashes, semantic decision digests, and final-chain commitments.

## 6. Static validation before expensive experiments

Reject a pilot before candidate search if any of the following holds:

- `B0 != 16N` under the declared baseline;
- 64× integer threshold is not `floor(N/4)`;
- duplicate paid-atom identities create double counting or ambiguity;
- a `DERIVED` dependency is cyclic or unresolved;
- tied-storage counting is inconsistent;
- model/decoder/quality IDs disagree across manifests;
- proof-critical artifacts lack content hashes.

Passing static validation does not establish compression feasibility. It only makes the experiment well-defined.

## 7. Candidate bit accounting

The serialized candidate must include every required paid component, including where applicable:

- roots / operators / dictionaries;
- coefficients;
- selectors;
- topology / packetization descriptions;
- codebooks / entropy-model state unless external-fixed;
- support masks/descriptions;
- headers, alignment, termination, padding;
- private residuals;
- checksums when required by the deployed protocol.

Shared atoms are counted once by unique identity.

## 8. Real-claim readiness

A real-model compression claim is ready only when:

1. baseline boundary is frozen;
2. tensor/operator extraction is architecture-valid;
3. paid-atom DAG validates;
4. actual candidate serialization exists;
5. quality witness uses the same frozen model/data/metric contract.

This contract is a gate, not evidence that 64× is achievable.
