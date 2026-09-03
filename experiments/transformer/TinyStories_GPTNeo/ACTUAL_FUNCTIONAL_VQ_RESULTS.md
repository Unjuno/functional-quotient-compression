# Actual TinyStories GPT-Neo functional VQ results

This note records high-value **actual-model** experiments run locally from the uploaded 1M / 3M / 8M / 28M TinyStories GPT-Neo checkpoints. Experiments were executed in the container; GitHub is only the canonical result store.

## Scope

- Architecture family: GPT-Neo, 8 layers, 16 heads.
- Hidden sizes: 64 / 128 / 256 / 512 for the 1M / 3M / 8M / 28M checkpoints.
- Tokenizer and forward were reconstructed locally from the uploaded checkpoint, `vocab.json`, and `merges.txt`; no external model download was used.
- Main intervention primitive in this note: 32-scalar block VQ, usually `K=128` (7-bit block index), with one shared Euclidean codebook per tensor role unless stated otherwise.
- Reported KL is next-token KL against the unmodified model on fixed TinyStories-style probes.
- Centroid serialization precision and final codec headers are **not** fixed in these experiments, so these results are task-geometry / representation evidence, not emitted-byte compression certificates.

## 1. Joint task distortion is not additive

For the 8M model, all 256 subsets of the eight WQ-layer interventions were evaluated exactly on a fixed calibration probe.

### Additive failure

For three compressed WQ layers:

- additive individual-KL model selected layers `(1,2,3)`;
- actual joint-KL oracle selected `(0,1,3)`;
- additive choice had `5.51%` joint-KL regret on the calibration probe.

On six unseen passages, `(0,1,3)` beat `(1,2,3)` in `6/6` cases:

- additive-choice mean KL: `0.01745`;
- joint-oracle mean KL: `0.01355`;
- holdout reduction: about `22.3%`.

For two compressed WQ layers, the calibration additive choice `(2,3)` also differed from the actual pair oracle `(1,3)`. On six holdouts, `(1,3)` won `5/6` and reduced mean KL from `0.01003` to `0.00779` (about `22.3%`).

### Pairwise correction

Define a measured pair interaction

`I(i,j) = D({i,j}) - D({i}) - D({j})`.

Using `sum individual + sum pair interactions`:

- triple prediction median relative error fell from `14.25%` to `0.55%`;
- triple ranking Spearman rose from `0.966` to `0.99945`;
- the actual oracle triplet was recovered.

For all 70 four-layer subsets:

- additive median error: `19.1%`;
- pairwise-corrected median error: `1.12%`;
- pairwise model recovered the exact four-layer oracle.

Across the complete 8-layer subset lattice, pairwise-corrected median error by subset size was approximately:

| compressed WQ layers | median relative error |
|---:|---:|
| 3 | 0.55% |
| 4 | 1.12% |
| 5 | 2.11% |
| 6 | 3.29% |
| 7 | 4.29% |
| 8 | 5.97% |

The pairwise model recovered the exact oracle subset at every cardinality `2..7`. The additive model missed at cardinalities `2`, `3`, and `5`.

Exact Möbius interaction magnitudes decayed quickly with order in this WQ experiment:

- order 2 mean absolute term: `2.02e-3`;
- order 3: `2.39e-4`;
- order 4: `3.90e-5`;
- order 5: `7.30e-6`.

This is strong evidence for a **pairwise-dominant functional coupling model**, not a proof that higher-order terms can always be ignored.

## 2. Pairwise correction transfers across scale and modules

Triplet experiments were repeated on other model sizes and roles.

### WQ scaling

- 1M: additive triplet choice missed the oracle with `11.0%` regret; pairwise correction recovered it; median pairwise prediction error `0.51%`.
- 3M: additive happened to choose the oracle but median prediction error was `12.8%`; pairwise correction reduced it to `0.54%`.
- 28M: additive median prediction error `4.50%`; pairwise correction `0.30%`; oracle recovered.

Pair interactions were not uniformly superadditive. Their sign and magnitude changed with model scale. The stable observation is that **measuring pairwise functional coupling was highly predictive despite those changes**.

### Other modules on 8M

- WK: additive triplet selection had `15.6%` regret; pairwise correction recovered the oracle; median prediction error `0.42%`.
- MLP `c_fc`: additive triplet selection had `8.9%` regret; pairwise correction recovered the oracle; median prediction error `3.78%`.

Higher-order effects are materially larger in the MLP case than in WQ/WK.

A six-candidate cross-role landscape using `Q1,Q3,K2,K3,V2,O2` was also exhaustively evaluated. At cardinality 2, additive selection had `49.3%` regret; pairwise correction recovered the oracle. Pairwise prediction error increased with subset size, reaching about `8.4%` for all six interventions.

## 3. Sub-bit rate tolerance is strongly role-dependent

At layer 2 of the 8M model, shared-role block-VQ codebooks were swept across `K=16,32,64,128,256` (index rates `0.125` to `0.25` bit/scalar before codebook state).

Weight NMSE improved monotonically with K for every tested role. Task KL did not.

Representative results:

- Q/K: task KL generally improves as K increases.
- V: across three codebook seeds, mean KL at `K64/K128/K256` was `0.1163 / 0.1118 / 0.1219` despite monotonically improving weight NMSE.
- O: task KL improves mildly with K.
- MLP `c_fc`: mean KL remained very large, about `0.392 / 0.382 / 0.369` at `K64/128/256`.

Thus a precision allocator must not use Euclidean reconstruction error as the task distortion curve for all roles. V and MLP are especially problematic under this primitive.

## 4. Task-aware assignment helps selectively, not universally

With the same shared K128 codebook and the same 7-bit index count, only block-to-centroid assignment was changed using calibration gradient-squared (diagonal Fisher-like) weights.

On six holdout passages at layer 2, `p=1` task weighting changed KL versus Euclidean assignment by approximately:

- Q: `-15.8%`;
- V: `-11.0%`;
- O: `-2.4%`;
- K: `+6.0%` (worse);
- MLP `c_fc`: `+2.2%` (worse).

Weight NMSE worsened in every role, confirming that better functional behavior need not imply better parameter reconstruction.

A nested exponent experiment used assignment weights `importance^p`, selecting p on three development passages and evaluating three unseen passages:

- Q selected `p=0.75` and improved test KL by `23.6%`;
- V selected `p=0.25` and improved by `27.8%`;
- K / O / MLP did not obtain a transferable improvement.

Therefore a task-aware metric itself requires **role-specific validation**. One universal diagonal-Fisher rule is unsafe.

## 5. Task-priced codebook coalitions beat naive grouping

Eight layer-local codebooks reduce reconstruction NMSE, but they pay eight fixed codebook costs. A single shared role codebook pays one fixed cost but can increase task distortion. The useful operating point is a discrete coalition problem.

A naive contiguous four-codebook partition (`[0,1],[2,3],[4,5],[6,7]`) was tested first. On six holdouts it was worse than eight local codebooks:

- Q: about `2.0%` higher mean KL;
- K: about `7.5%` higher mean KL.

The improvement seen on the calibration probe therefore did **not** transfer; contiguous grouping is not a valid coalition rule.

### Actual pair-priced matching

For each role, all 28 layer pairs were trained with one shared K128 codebook. The actual calibration task cost of merging each pair was measured. All 105 perfect matchings of eight layers into four pair-codebooks were then enumerated exactly.

#### Q role

Holdout mean single-intervention KL across the eight layers:

- contiguous four-codebook partition: `0.010574`;
- eight local codebooks: `0.010366`;
- **task-priced four-codebook matching: `0.009606`**.

The task-priced matching therefore uses **half as many codebooks** while achieving about **7.3% lower holdout KL than local8**.

#### K role

- contiguous four-codebook partition: `0.010599`;
- eight local codebooks: `0.009862`;
- **task-priced four-codebook matching: `0.009713`**.

Again, half as many codebooks are used, with about **1.5% lower holdout KL than local8**.

This is actual Transformer evidence that **shared-representation fixed costs should be optimized as task-priced coalitions, not inferred from layer adjacency or raw weight similarity alone**.

## 6. Implication for QCO

The strongest optimizer architecture supported by these experiments is now:

1. generate discrete representation/rate actions;
2. estimate individual functional cost;
3. measure selected pairwise functional couplings;
4. price shared-codebook/root coalitions using actual merge cost;
5. allow replacement / regrouping moves;
6. use exact full replay for final commit.

The current evidence argues against:

- additive individual task costs;
- raw reconstruction NMSE as the universal task metric;
- one shared codebook for every role;
- layer-neighbor grouping as a coalition heuristic;
- one universal Fisher weighting rule.

## Boundaries

Do not infer that:

- K128 is a viable full-model codec for V/O/MLP;
- centroid values have already been quantized and serialized at the claimed model-wide bit budget;
- pairwise correction is globally exact for arbitrary codec actions;
- the six short holdouts constitute benchmark-quality language-model evaluation;
- these TinyStories results prove 64x feasibility on SmolLM2 or any larger Transformer.

The result is narrower: **actual Transformer task geometry is measurably nonadditive, largely pairwise-dominant in the tested attention landscapes, and shared-codebook coalitions can be improved by task-based pricing enough to beat local-codebook holdout quality at lower fixed representation count.**