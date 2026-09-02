# T001 — SmolLM2-135M Real-Model Pilot

This directory is the first real-model FQC pilot. The pinned source checkpoint
has passed the **serialized checkpoint gate**, the **live runtime / layer-0
replay gate**, and the first **task-unconditioned raw structural audit**. The next
required phase is task-conditioned functional sensitivity.

## Pinned source

- model: `HuggingFaceTB/SmolLM2-135M`
- immutable Hub snapshot: `28e66ca6931668447a3bac213f23d990ad3b0e2b`
- checkpoint: `model.safetensors`
- checkpoint SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- checkpoint size: 269,060,552 bytes

Machine-readable evidence lives under `results/`.

## G1 — actual serialized checkpoint: PASS

GitHub Actions downloaded the immutable checkpoint and verified:

- serialized tensors: **272**
- serialized scalars: **134,515,008**
- dtype inventory: **134,515,008 BF16 scalars**
- missing checkpoint keys: **0**
- unaccounted checkpoint keys: **0**
- safetensors header: **30,528 bytes**
- 8-byte prefix + header: **30,536 bytes**
- tensor payload: **269,030,016 bytes**
- total file: **269,060,552 bytes**

The entire difference between the BF16 tensor payload and file size is exactly
the safetensors length prefix plus JSON header.

## Baseline denominator and 64x target

The pinned serialized source checkpoint is the authority for the paid baseline
scalar payload:

`N = 134,515,008`

Under the project 16-bit baseline convention:

- baseline payload: 2,152,240,128 bits = 269,030,016 bytes
- hard 64x budget: **33,628,752 bits = 4,203,594 bytes**

Runtime loader duplication must not inflate this denominator.

## G2 — live runtime extraction and layer-0 replay: PASS

The model was loaded from the pinned files with:

- Transformers 4.47.1
- Torch 2.5.1+cpu
- CPU
- BF16
- eager attention
- fixed seed 20260902

The runtime manifest reproduced `N = 134,515,008`. The tied token embedding and
LM head were both the same Parameter object and the same underlying storage.
The only runtime state-dict key not serialized separately in the source was the
expected tied alias `lm_head.weight`.

Five independently reconstructed layer-0 cases passed the predeclared
`atol=rtol=0.02` replay contract:

| case | max absolute error | reference/extracted hash |
|---|---:|---|
| input RMSNorm | 0.0 | identical |
| RoPE cos/sin | 0.0 | identical |
| eager GQA attention output | 0.0 | identical |
| SwiGLU MLP output | 0.0 | identical |
| full decoder layer | 0.0 | identical |

A second fresh GitHub runner reproduced the same runtime manifest canonical hash
and all five replay hashes. The explicit Llama RoPE contract is
`concat_freqs_freqs + half_split_rotate_half`.

This proves extraction/runtime semantics for the tested layer and runtime; it is
**not** compression evidence.

## Immediate 64x budget constraint

The actual denominator makes a useful hard arithmetic constraint visible:

- tied token embedding / LM head: **28,311,552 scalars = 21.05% of N**
- all attention weights: 26,542,080 scalars = 19.73% of N
- all MLP weights: **79,626,240 scalars = 59.20% of N**

If the entire 64x budget were spent on the embedding alone, it could use at most
**1.188 bits per embedding scalar**. A 2-bit embedding by itself would exceed the
entire model budget by **22,994,352 bits = 2,874,294 bytes**, even if every other
component cost zero. At 1 bit per embedding scalar, the other 106,203,456 scalars
would have only 5,317,200 bits left, or **0.0501 bit/scalar on average**.

These are arithmetic constraints, not quality or codec lower bounds. They show
that a 64x result cannot come from ordinary low-bit treatment of the rest of the
network while leaving the embedding conventionally quantized; embedding and/or
other dominant structures must obtain very low description length.

See `results/source_component_budget.json`.

## G3a / T002 — raw structural audit: PASS

Layers **0, 15, and 29** were audited directly from the pinned BF16 checkpoint.
The full result was reproduced byte-for-byte across two independent GitHub
Actions runs (`structural_audit_result.json` SHA-256
`b051c3b4b87430902156451ccee11d6b166260c77b839f4f2f850dda56fa6a77`).

The descriptive reference is an equal-energy mutually orthogonal family: for
`m` Frobenius-normalized family members its rank-1 residual is `1 - 1/m`. This is
**not** a statistical null test.

### Main findings

1. **Direct raw cross-layer sharing is essentially absent.** For corresponding
   WQ/WK/WV/WO/gate/up/down tensors across layers 0/15/29, the largest rank-1
   concentration excess above the orthogonal reference is only **0.00225**
   (WQ). Pairwise raw-weight cosines are correspondingly near zero.
2. **Local GQA Q/K families are much more structured.** The strongest audited
   family is layer 15, KV group 2 (`3 Q heads + corresponding K head`): rank-1
   residual **0.4731** versus orthogonal reference **0.75**, an excess
   concentration of **0.2769**. Strong candidates also appear in layer 29.
3. **Value-output derived operators show moderate concentration**, especially
   layer 29 (excess **0.1662**) and layer 15 (**0.1404**). Because GQA already
   shares V parameters, this must not be double-counted as new compression gain.
4. **MLP channel structure is layer-dependent.** In layer 15 the top 10% of
   SwiGLU channel descriptor energy contains **23.73%** of the total and the top
   25% contains **44.72%**. Layer 0 is much closer to uniform (top 10% =
   **11.75%**).

The practical consequence is that root search should prioritize **local,
architecture-aware families** rather than a single shared raw basis across
distant layers. A uniform per-layer codec policy is also poorly motivated by
these measurements.

See `results/structural_audit_summary.json`. Full exploratory matrices and
singular spectra remain in the GitHub Actions artifact rather than the canonical
repository result.

## Evidence boundary

The current T001/T002 evidence establishes:

- exact pinned source identity and denominator,
- full serialized tensor coverage,
- runtime tied-storage consistency,
- deterministic adapter semantics,
- exact layer-0 replay under the pinned runtime,
- reproducible raw structural measurements on layers 0/15/29.

It does **not** yet establish:

- task-weighted null directions,
- functional sensitivity or removability of structural candidates,
- actual root/dictionary bit savings,
- any actual FQC encoded byte count,
- or 64x feasibility.

## Next gate — G3b functional audit

1. Fix a calibration corpus and hash the exact tokenized samples.
2. Capture layer inputs / outputs under the pinned runtime.
3. Measure D61-style functional sensitivity for the strongest T002 candidates,
   especially layer-15/29 GQA groups and layer-15 MLP channels.
4. Compare structural score against functional score; explicitly search for
   low-energy/high-task-value counterexamples.
5. Only then generate shared-root/private-residual codec candidates and price
   them against the **4,203,594-byte** hard budget.
