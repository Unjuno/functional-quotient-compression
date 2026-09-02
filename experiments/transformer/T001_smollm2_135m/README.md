# T001 — SmolLM2-135M Real-Model Pilot

This directory is the first real-model FQC pilot. The pinned source checkpoint
has passed the **serialized checkpoint gate**, **live runtime / layer-0 replay**,
**raw structural audit**, and the first **task-conditioned functional
intervention audit**. The next research question is why structurally similar
candidates can differ greatly in functional sensitivity.

## Pinned source

- model: `HuggingFaceTB/SmolLM2-135M`
- immutable Hub snapshot: `28e66ca6931668447a3bac213f23d990ad3b0e2b`
- checkpoint: `model.safetensors`
- checkpoint SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- checkpoint size: 269,060,552 bytes

Machine-readable evidence lives under `results/`. Full exploratory output remains
in GitHub Actions artifacts unless it is needed for future reproduction.

## G1 — actual serialized checkpoint: PASS

The immutable checkpoint establishes:

- serialized tensors: **272**
- paid BF16 scalars: **134,515,008**
- missing/unaccounted checkpoint keys: **0 / 0**
- tensor payload: **269,030,016 bytes**
- safetensors prefix + header: **30,536 bytes**
- total file: **269,060,552 bytes**

The pinned serialized source checkpoint is the authority for the paid baseline
scalar payload: `N = 134,515,008`.

Under the 16-bit baseline convention:

- baseline payload: 2,152,240,128 bits = 269,030,016 bytes
- hard 64x budget: **33,628,752 bits = 4,203,594 bytes**

## G2 — live runtime extraction and layer-0 replay: PASS

Pinned runtime: Transformers 4.47.1, Torch 2.5.1+cpu, CPU BF16, eager attention,
seed 20260902. Runtime storage reproduced `N`; token embedding and LM head were
the same Parameter and storage.

Independent reconstructions of input RMSNorm, RoPE, eager GQA attention, SwiGLU
MLP, and the full decoder layer all had **maximum absolute error 0.0** with
identical reference/reconstructed hashes. A fresh runner reproduced the same
runtime manifest and replay hashes. The explicit RoPE contract is
`concat_freqs_freqs + half_split_rotate_half`.

This certifies extraction semantics for the tested runtime; it is not compression
evidence.

## Immediate 64x budget constraint

- tied embedding / LM head: **28,311,552 scalars = 21.05% of N**
- attention weights: 26,542,080 scalars = 19.73%
- MLP weights: **79,626,240 scalars = 59.20%**

If the entire 64x budget were spent on the embedding alone, it could use at most
**1.188 bits/scalar**. A 2-bit embedding alone exceeds the whole model budget by
**22,994,352 bits = 2,874,294 bytes**. Even at 1 bit/embedding scalar, the other
106,203,456 scalars would have only **0.0501 bit/scalar** on average.

These are arithmetic constraints, not codec or quality lower bounds.

## G3a / T002 — raw structural audit: PASS

Layers **0, 15, and 29** were audited from the pinned checkpoint. Two independent
runs produced byte-identical full results (SHA-256
`b051c3b4b87430902156451ccee11d6b166260c77b839f4f2f850dda56fa6a77`).

Main findings:

1. **Direct raw cross-layer sharing is essentially absent.** Corresponding
   WQ/WK/WV/WO/gate/up/down across layers 0/15/29 are near the equal-energy
   orthogonal reference; the largest rank-1 concentration excess is only
   **0.00225**.
2. **Local GQA Q/K families are much more structured.** The strongest audited
   family is layer 15, KV group 2 (`3 Q heads + corresponding K head`): rank-1
   residual **0.4731** versus orthogonal reference **0.75**.
3. **Derived V→O operator families show moderate concentration**, especially
   layer 29 and layer 15, but known GQA reuse must not be double-counted as new
   rate gain.
4. **SwiGLU structure is layer-dependent.** Layer 15 top 10% of channel
   descriptor energy holds **23.73%**; layer 0 top 10% holds only **11.75%**.

This makes local architecture-aware root search more plausible than one shared
raw basis across distant layers.

## G3b / T003 — functional intervention audit: PASS

A fixed four-passage authored calibration set (164 valid next-token positions)
was tokenized under the pinned tokenizer. It is **not a benchmark**; it is a
reproducible activation distribution for sensitivity ranking. Two fresh-runner
executions produced byte-identical full results (SHA-256
`e43343ffd93085e52b5f2611d7d02fab7e1f55321cbbdf2feb369326d74a578a`).

### Structural residual is not a functional metric

The clearest comparison is the same rank-3 GQA intervention at two layers:

| candidate | structural residual | mean KL | top-1 flip |
|---|---:|---:|---:|
| L15 KV2 rank-3 | 0.09015 | **0.03013** | **7.32%** |
| L29 KV2 rank-3 | 0.10745 | **0.001404** | **0.61%** |

The structural residuals are similar, but layer 15 has **21.46x larger KL** and
**12x larger top-1 flip rate**. Raw family geometry alone therefore does not
specify task sensitivity; activation distribution, layer location, and downstream
continuation gain are candidate explanations.

Within the same layer-15 family, increasing structural residual from **0.0902**
to **0.4731** (5.25x) raises KL only from **0.0301** to **0.0358** (1.19x). The
relationship is not calibrated even within one family.

### Calibration NLL alone is unsafe as an admission metric

All three layer-15 Q/K approximations have a *negative* next-token NLL delta on
this tiny calibration set, while still producing KL around 0.03 and 7–10% top-1
changes. This is not evidence that the perturbed model is better; it is a direct
example of why a single calibration loss cannot authorize a codec move.

### Low descriptor score is not task irrelevance

Removing the bottom ~10% of layer-15 SwiGLU channels by the T002 descriptor
removes only **3.24%** of descriptor energy, yet causes:

- mean KL **0.01175**
- top-1 flip **6.10%**
- calibration NLL delta **+0.01247**

Those channels still contain about 10% of the layer's parameter Frobenius energy,
so this is not a conventional low-weight-energy pruning result. It nevertheless
shows that the structural descriptor by itself is not a removability certificate.

### A promising local candidate, not yet a codec result

`L29_qk_kv2_rank3` is the least sensitive tested Q/K intervention:

- structural residual 0.10745
- mean KL **0.001404**
- top-1 flip **0.61%**
- relative RMS logit delta **0.00724**

But one rank-3 4-member Q/K family affects only 147,456 raw scalars. Even a naive
continuous rank-3 parameterization would save only about **36,848 scalar
coefficients**, roughly **0.0274% of model N**, before quantization and metadata.
Repeated reuse or a much broader quotient is therefore required for material
model-level compression.

See `results/functional_audit_summary.json`.

## Evidence boundary

The current real-model evidence establishes:

- exact source identity and baseline denominator,
- full serialized tensor coverage,
- runtime tied-storage consistency,
- exact layer-0 extraction replay,
- reproducible raw structural measurements,
- reproducible functional sensitivity of a small predeclared intervention set.

It does **not** establish:

- benchmark/generalization quality under perturbation,
- a global task-null dimension,
- codec-positive root/dictionary moves,
- actual FQC encoded bytes,
- or 64x feasibility.

## Next gate — T004 local sensitivity vs continuation gain

The immediate question is why structurally similar layer-15 and layer-29
rank-3 GQA interventions differ by about **21.5x in KL**.

The next experiment should separate:

1. activation-weighted local layer-output perturbation,
2. downstream continuation amplification from that layer to final logits,
3. raw structural residual.

A scalable screen can then evaluate all 90 GQA groups locally and reserve full
model validation for a small set of candidates. Only after this task-conditioned
screen should root/dictionary candidates be priced against the
**4,203,594-byte** hard budget.
