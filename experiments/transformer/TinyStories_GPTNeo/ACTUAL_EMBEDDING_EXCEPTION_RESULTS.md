# Actual TinyStories GPT-Neo embedding exception results

This note records high-value **local real-model** experiments performed on user-provided TinyStories/GPT-Neo checkpoints. These results are separate from the pinned SmolLM2 lane. They are not a 64x model certificate.

## Execution boundary

- Models: user-provided GPT-Neo checkpoints at hidden sizes 64/128/256/512; provenance hashes are recorded in `LOCAL_CHECKPOINT_PROVENANCE.json`.
- Main detailed experiments below use the hidden-size-256 / `8M` checkpoint.
- Forward path: local PyTorch implementation of GPT-Neo causal attention, MLP, layer norms, tied input embedding / LM head, and GPT-2 byte-level BPE from the supplied tokenizer files.
- Probe lengths were kept below the configured local-attention window, so global/local attention have the same causal receptive field on these probes.
- The local tokenizer reproduces standard GPT-2 examples (`Hello -> 15496`, `Once upon a time -> [7454,2402,257,640]`).
- An independent Hugging Face forward replay was not available in this container; therefore this lane is local real-model evidence, not a replacement for the stricter SmolLM2 extraction certificate.

## 1. Hard-budget representation studied

The main embedding experiments use a component budget of approximately `0.25 bit/scalar` and a representation with:

1. a task-signature root codebook;
2. decoder-visible root assignments for every vocabulary token;
3. sparse high-importance full-dimensional residual exceptions;
4. explicit selector, per-row quantization scale, codebook, and header accounting.

Functional importance is computed from calibration gradients for both the output/LM-head role and the input-embedding role. A combined normalized importance score was consistently stronger than input-only or output-only selection in the latest local experiments.

## 2. Sparse full-dimensional exceptions beat diffuse private corrections

At the same approximately `0.25 bit/scalar` budget on the 8M checkpoint:

- root + sparse 4-bit full residual exceptions achieved six-holdout mean KL around `0.49–0.60` depending on root configuration and probe split;
- adding a low-dimensional residual stream to every token consumed budget, reduced the number of full exceptions, and worsened KL;
- a global residual-PCA dictionary could improve ordinary weight NMSE while producing catastrophic functional behavior: fresh-holdout KL remained roughly `2.4–3.3` and top-1 flips approached `95–99%` in many configurations;
- broad Euclidean residual-VQ stages similarly improved weight NMSE without improving KL over the sparse-only representation;
- a weak task-dimension-weighted residual-VQ stage improved over Euclidean broad VQ, but still did not beat sparse mixed-precision exceptions.

The practical conclusion is narrow but important: **under this extreme embedding budget, broad low-fidelity reconstruction is a poor substitute for sparse high-fidelity functional exceptions.**

## 3. Root complexity and private budget trade off

A root-size sweep showed that larger roots are not automatically better.

For one six-holdout sweep using sqrt-importance-weighted centroids and uniform 4-bit exceptions:

| root K | mean KL |
| ---: | ---: |
| 8 | 0.602 |
| 16 | 0.580 |
| 32 | 0.570 |
| 48 | 0.578 |
| 64 | 0.567 |
| 96 | **0.564** |

On a separate three-text held-out split, `K=64` outperformed the development-selected `K=256` configuration (`0.249` vs `0.280` KL). The direction is consistent with a fixed-charge tradeoff: spending more bits on the shared root can reduce the number of private exceptions enough to hurt functional generalization.

## 4. Functional mixed precision transfers to fresh text

With a `K=96` root and the same hard component budget:

- uniform 4-bit residual exceptions: fresh six-text KL `0.49394`, top-1 flip `31.55%`;
- top 384 highest-importance exception tokens at 5-bit, remaining exceptions at 4-bit: fresh KL **`0.46650`**, top-1 flip **`22.13%`**;
- top 256 tokens at 6-bit, remaining at 4-bit: fresh KL `0.47207`, top-1 flip `22.28%`.

Thus a small amount of precision reallocation toward high-functional-leverage tokens improved both KL and decision stability without increasing the bit budget.

Hidden-dimension-level mixed precision gave smaller gains. Combining token-level and hidden-dimension-level precision did not materially improve KL beyond token mixed precision. The dominant allocation coordinate in these experiments is therefore the **token axis**, not the hidden-dimension axis.

## 5. Better weight reconstruction can be much worse functionally

The strongest negative result in this batch is that lower parameter reconstruction error is not a reliable codec objective.

A residual-PCA representation could give lower embedding NMSE than the sparse exception representation while producing several-fold larger KL and nearly complete top-1 disagreement. Broad residual-VQ stages showed the same direction: extra reconstruction stages lowered NMSE while functional quality worsened after the first stage.

This independently reproduces the same research boundary previously observed in SmolLM2 Q/K interventions: **parameter-space error and functional distortion are different geometries.**

## 6. Current best interpretation

For the tested 8M tied embedding at approximately `0.25 bit/scalar`, the current ordering is:

`task-signature root + sparse functional exceptions + token mixed precision`

better than

`task-aware broad residual VQ + sparse exceptions`

better than

`Euclidean broad residual VQ`

better than

`global low-rank residual reconstruction`.

Even the best current local configuration still has substantial KL, so this is **not evidence that the embedding has been solved at a 64x-equivalent component rate**. It is evidence about where the bit budget should be concentrated.

## Boundaries

Do not infer that:

- the local GPT-Neo manual forward is an independent extraction certificate;
- these hand-written calibration/holdout probes are benchmark-quality evaluation;
- a `0.25 bit/scalar` embedding component budget is sufficient for acceptable deployed quality;
- individual component results add linearly when the whole model is compressed;
- the root/exception parameters reported here are universal across model scales or architectures.

The next high-value experiment is a broader held-out functional evaluation of the sparse-exception codec, followed by actual non-embedding compression under the same serialized model-wide budget.
