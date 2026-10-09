# MA-461 — HyperFormer generator outputs Mirror codes

Status: **FAIL**  
Evidence lane: GENERATED ADAPTERS / QUALITY / GENERATOR BYTES / EXTRAPOLATION  
Protocol frozen: `682f9588`; fresh worlds 46110–46112.

## H — Hypothesis

A context generator that predicts two Mirror coordinates decoded through one shared 2×2 matrix can reproduce held-out task/layer/position adapters with no more than 10% NRMSE loss and at most 70% of the full HyperFormer-style generator payload.

## T — Test

Each world defines 2×2 adapter targets affine in a four-value context containing two task attributes, layer, and adapter position. HyperFormer control linearly generates all four adapter entries; Mirror predicts two angles and decodes them as `R(m₀) W R(m₁)`; rank-2 basis control uses a generic factorized linear generator. All receive the same 64 training contexts and are evaluated on 64 held-out contexts, with 64 support examples and 256 queries per context. Development selected 1,000 Mirror updates from 200/500/1,000. Three fresh worlds × three seeds; actual payload bytes include generator, decoder, normalization, context values, and metadata.

## D — FAIL

**Fact:** At N=20, mean NRMSE was 5.51e-7 for full HyperFormer, 0.908 for Mirror, 0.364 for rank-2 basis, and 0 for the teacher-matrix upper control. Actual bytes/context were 174.25B HyperFormer, 180.45B Mirror, 180.45B rank-2, and 222.25B independent. Mirror misses both the 10% quality and 70% byte gates. At equal serialized bytes, the generic rank-2 basis control is substantially more accurate. Mirror also used 26 generator-plus-decoder MACs/context-query versus 24 for full HyperFormer and had slower query wall in this CPU screen.

**Interpretation:** Replacing full adapter generation with two Givens coordinates greatly restricts the generated function family. The slight nominal parameter reduction did not reduce actual serialized bytes and gave poor held-out quality; a generic rank-2 basis was a much stronger compact control.

## C — Strongest counter-hypothesis

The affine-in-context teacher spans a general matrix family while the Mirror decoder is constrained to a two-angle orbit of one shared matrix. The result may reflect a mismatch between Mirror geometry and this adapter family rather than a universal limit on generated Mirror codes.

## U — Unknown

Transformer fine-tuning, learned task/layer embeddings, nonlinear HyperFormer generators, larger adapter ranks, and Mirror decoders matched to real adapter geometry remain untested.
