# Actual TinyStories-28M serialized embedding hierarchy results

This note records high-value **local real-model / embedding-only codec** experiments performed in the container on the user-provided `roneneldan/TinyStories-28M` checkpoint. It extends the earlier TinyStories embedding studies; it is not a whole-model 64x certificate.

## Provenance and execution boundary

- Model: `roneneldan/TinyStories-28M`, GPT-Neo, hidden size 512, 8 layers, vocabulary 50,257.
- User-provided ZIP SHA256: `d17e450912244aed8ee90933d8d91b3f8fd70e6a8f341eb3ce27b54fdfdf9958`.
- `pytorch_model.bin` SHA256: `8ddd260f51b439744c8cc785b5516327d4bf32e31ccbfa9009bfadf12557fcf5`.
- Functional replay uses the local PyTorch GPT-Neo implementation and supplied GPT-2 byte-level BPE files. Probe lengths are below the local-attention window.
- The embedding has `50,257 x 512 = 25,731,584` scalars, so a proportional `0.25 bit/scalar` embedding-component budget is exactly `804,112` bytes.
- This component budget is not a whole-model compression ratio.

## 1. Private-bit allocation: sparse beats diffuse

With a K256 task-signature root and a fixed 0.25-bps packed budget, broad low-dimensional corrections were inferior to sparse full-vector private corrections.

Representative four-holdout results:

- 2,629 4-bit full residual exceptions: mean KL `1.8595`;
- 26-dimensional hybrid global coefficients for essentially all tokens: mean KL about `2.39`;
- low-dimensional sparse exceptions could cover many more tokens but still did not beat the full-vector exception baseline.

PCA/global corrections could improve ordinary embedding NMSE while worsening task KL. The useful allocation variable is therefore not simply reconstructed parameter energy.

## 2. Task-consistent residual VQ helps, but does not replace the sparse core

A shared 32-dimensional residual block-VQ greatly improved weight NMSE but was functionally poor under Euclidean assignment.

For K128 under the same 0.25-bps accounting:

- Euclidean residual VQ: embedding NMSE `0.6357`, mean KL `3.2099`;
- calibration-hidden-covariance / Mahalanobis assignment with the same codebook and bits: NMSE worsened to `0.7269`, but mean KL improved to `2.7047`.

Thus task-consistent assignment moves error in the right functional direction, but broad VQ alone remains much weaker than sparse high-fidelity exceptions.

## 3. KL/Hessian-aware exception pricing is a major gain

The largest improvement in this batch came from changing **only which tokens receive full residual exceptions**.

The root, exception count (`2,629`), 4-bit residual format, and packed rate were unchanged.

- unweighted `delta_logit^2` ranking: mean KL `1.8595`, mean top-1 flip `62.0%`;
- softmax-probability / diagonal-Hessian ranking `p * delta_logit^2` or `p(1-p) * delta_logit^2`: mean KL **`0.7024`**, mean top-1 flip **`33.5%`**.

Embedding NMSE became slightly worse (`0.704 -> 0.711`) while task quality improved dramatically. This is direct actual-model evidence that private bits should be priced in functional geometry, not by reconstruction error.

A naive individually decomposed quadratic-marginal score performed catastrophically (mean KL about `3.61`). Softmax-coupled distortion is therefore not safely reducible to that independent-token marginal formula.

## 4. Hierarchical private representation improves further

The functionally priced sparse core was combined with a lower-precision, task-aware shared residual-VQ tail.

At the same approximately 0.25-bps component budget:

- p-weighted full exceptions only: mean KL `0.7024` on the original four holdouts;
- K256 / 2-bit tail + 2,250 full exceptions + 5,341 VQ-tail tokens: mean KL `0.5646`;
- K512 / 2-bit tail + 2,250 full exceptions + 4,679 VQ-tail tokens: mean KL `0.5603`.

The K512 gain over K256 is small; most of the benefit comes from the hierarchy itself:

`shared root -> high-fidelity sparse private core -> low-bit task-aware private tail`.

Tail codebook precision is not monotone in task quality. Across three K128 codebook-training seeds, 2-bit tail codebooks averaged KL `1.576`, compared with `1.584` for 3-bit and `1.599` for 4-bit, even though higher precision improved parameter NMSE.

## 5. Transfer to additional unseen passages

Without retraining the root, selector, or tail codebook, the K512 hierarchy was evaluated on eight additional unseen TinyStories-style passages.

Mean KL:

| Evaluation set | p-weighted full exceptions | hierarchical private |
| --- | ---: | ---: |
| original 4 | 0.7024 | **0.5603** |
| new 8 | 0.8692 | **0.7532** |
| all 12 | 0.8136 | **0.6889** |

The hierarchy improved KL on **12/12 passages**. Absolute distortion remains large, so this is evidence for the representation hierarchy, not evidence that 0.25-bps embedding quality is acceptable.

## 6. Root candidate generation is functionally unstable

K256 task-signature KMeans roots can have nearly identical reconstruction statistics while materially different functional cost.

Across 10 root seeds in a root-only screen:

- KMeans inertia vs holdout KL Spearman: `0.564`;
- embedding NMSE vs holdout KL: `0.285`;
- **frozen-hidden output KL vs holdout KL: `0.879`**;
- local input-embedding error vs holdout KL: `0.806`.

The frozen functional surrogate uses baseline final hidden states and applies the candidate decoded embedding/LM-head without a full-model candidate replay. It is therefore a substantially better proposal-ranking signal than KMeans objective or weight NMSE in this screen. Exact functional replay remains the commit authority.

## 7. Actual serialized embedding-only file

The best hierarchy was serialized to a deterministic binary format with serializer-consistent FP16 scales.

Fields include:

- 4 KiB header;
- K256 root labels;
- 4-bit root values and FP16 per-root scales;
- 2,250 uint16 full-exception token IDs;
- 4-bit full residuals and FP16 scales;
- K512 2-bit shared tail codebook and FP16 scales;
- 4,679 uint16 tail token IDs;
- 9-bit packed tail block indices.

Results:

- actual file size: **`804,101` bytes**;
- 0.25-bps component limit: **`804,112` bytes**;
- headroom: **11 bytes**;
- binary SHA256: `c8af91de2df4d061a2b12c7588901ca6a69aab8cfccb11870e491c533783fb6b`;
- decoder round-trip vs encoder-side decoded embedding: **max absolute difference `0.0`**;
- 12-passage mean KL after decoding: **`0.66993`**;
- mean top-1 flip: approximately **37.3%**.

This is an **embedding-only serialized rate PASS and quality FAIL**. It does not establish a whole-model codec, a 64x model result, or acceptable task quality. The 11-byte headroom also shows that this exact layout is serializer-fragile.

## Scientific interpretation

The current actual-model evidence supports a narrower FQC architecture:

`functionally priced shared root -> high-leverage full private exceptions -> task-aware low-bit shared private tail -> exact serializer -> exact functional replay`.

The dominant improvement came from allocating private bits according to softmax/task geometry, not from lowering weight reconstruction error. Root identity, private-core size, tail cardinality, and tail precision are discrete joint variables; reconstruction-optimal choices are not reliable commit choices.

## Boundaries

Do not infer that:

- the TinyStories manual forward is an independent Hugging Face extraction certificate;
- 0.25-bps embedding quality is acceptable;
- embedding-only rate success implies whole-model 64x feasibility;
- the K256/K512 settings or 2-bit tail precision generalize to SmolLM2;
- individual component distortions add under simultaneous whole-model compression.
