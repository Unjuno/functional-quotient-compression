> **HISTORICAL SNAPSHOT — superseded as current-state entry point by [CURRENT_STATE_2026-10-07.md](CURRENT_STATE_2026-10-07.md).**\n> Experimental findings remain valid within their original scope.\n\n# Research state — Mirror Transformer through 2026-10-06

Date: 2026-10-06  
Status: **integrated evidence summary**

This document consolidates the experimental record from Phase I FQC through MN, MT, router-free RF, and reachability-adjusted work. It preserves negative and mixed results. Proposed token/sensor Mirror extensions are listed separately from measured evidence.

## 1. Executive conclusion

The project has not yet demonstrated a reproducible Mirror-specific capacity multiplier.

The strongest current conclusions are narrower:

1. functional reuse can be trained;
2. a wide shared computation can be competitive with duplicated narrow experts;
3. Mirror **breadth** matters separately from Mirror **count**;
4. the tested diagonal/gain and narrow rotation families are too limited to support a capacity claim;
5. multiple deterministic views can improve robustness to unseen views;
6. packed multi-view forward/backward is numerically valid and can reduce loop overhead;
7. choosing Mirror directions only by local reachability cost does not reliably improve subsequent shared-weight learning;
8. the next architecture should use low-description non-rotation transforms and evaluate token/sensor observation coordinates.

## 2. Phase I / FQC boundary

The original FQC objective was post-training functional compression: identify degrees of freedom equivalent under a task, share them, and verify actual serialized bytes under a quality constraint.

Important negative result:

- on TinyStories 28M/8M/1M in the tested sharing/codebook family, sharing/private variants did not beat strong activation-aware non-sharing controls in viable regimes.

Interpretation:

- invertible transforms alone do not create compression;
- apparent shared-state holdout wins may come from regularization/variance reduction;
- task-relevant cross-fitted private energy is more informative than ambient width/rank alone;
- this motivates designing sharing into training rather than assuming naive post-training sharing will work.

## 3. MN007-MN010: reusable factors and global Mirror breadth

### MN007

Shared semantic bases can be compact, but spontaneous grouping was weak and additive/low-rank controls were strong.

### MN008

Explicit functional recombination supervision made reusable factors substantially better:

- held-out recombination improved strongly in both the main and fresh world;
- semantic grouping improved;
- additive/shared-base controls also benefited and remained strong.

Conclusion: **factor reuse is trainable, not Mirror-specific.**

### MN009

Compared Dense, simple gate, shared Mirror FFN, near-byte small soft-MoE, and larger soft-MoE.

Measured storage:

- Mirror: 86,127 B;
- small independent soft-MoE: 85,665 B;
- large independent soft-MoE: 181,367 B.

Mirror beat the near-byte narrow soft-MoE in paired IID/OOD NLL, but Dense/simple-gate controls remained competitive or better and Mirror did not match the larger MoE.

Conclusion: preserving a wide shared path can be better than fragmenting an equal storage budget into narrow experts, but no general Mirror-specific superiority was shown.

### MN010

Separated Mirror count from Mirror breadth.

- increasing low-dimensional state count E=1->8 did little;
- a broad 256-scalar global state produced a discovery-world signal;
- the fresh-world replication was mixed.

Conclusion: **count alone is insufficient; functional breadth per state matters.**

See [RESEARCH_STATE_THROUGH_MN010.md](RESEARCH_STATE_THROUGH_MN010.md).

## 4. MT001: multi-state paired training

117 train/audit models were run in the container study.

Main observations:

- paired multi-state training reduced held-out NLL in the original comparison;
- clean/IID tolerance was not met;
- held-out accuracy remained low;
- same-state repeated-input controls reproduced much of the NLL pattern;
- high gradient agreement could coexist with poor task accuracy.

Therefore the result did **not** establish shared-rule acquisition or capacity growth.

Engineering result:

- four-view packed/shared-prefix execution was numerically equivalent to loop execution on the tested fixture;
- the tiny CPU benchmark showed about 2.22x speedup versus the four-view Python loop for the shared-prefix implementation.

Interpretation: multi-view execution is feasible, but repeated exposure/optimization effects must be controlled.

## 5. MT002A/B: routing versus representation

### MT002A — learned routing

The learned-routing Mirror bridge failed the cross-world gate:

- one world showed a positive paired OOD-NLL signal;
- the other world was negative in all three paired seeds.

Conclusion: learned routing was not robust enough.

### MT002B — oracle routing

Giving the correct primitive-rule identity improved all conditional models, but the gain was much larger for independent experts.

Median oracle OOD accuracy included:

- Mirror: roughly 12.5% / 15.0% in the two worlds;
- small independent Full-MoE: roughly 53.9% / 75.3%.

Conclusion: routing was a bottleneck, but **Mirror representation breadth was also a bottleneck**.

## 6. MT003A: Mirror breadth sweep

At approximately matched storage, widening the tested gain-Mirror family did not yield a monotonic law.

Representative medians:

- best Mirror OOD NLL: about 6.90;
- best Mirror OOD accuracy: about 25.6%;
- comparable-storage oracle Full-MoE: OOD NLL about 1.89 and accuracy about 62.3%.

Conclusion: shared-feature + diagonal/gain modulation is too narrow as the principal expert parameterization.

## 7. RF1: router-free deterministic Mirror

A later router-free experiment removed learned routing and used deterministic fixed rotations around the FFN nonlinearity.

Scope:

- 12 layers;
- hidden width 64;
- FFN width 128;
- 4 heads;
- 438,912 learned parameters in all compared models;
- P=10 layer period;
- K=8 rotation views;
- rho=0.2 rad;
- 6 conditions x 2 task families x 3 training seeds = 36 trained/audited models;
- 4,000 AdamW updates per run.

### Quality result

K=8 rotation Mirror did **not** show a stable Dense advantage on held-out rule composition.

Paired OOD-accuracy gains versus Dense:

- mod31 family: median about -0.48 percentage points; positive in 1/3 seeds;
- GF(2) family: median about -2.03 percentage points; positive in 1/3 seeds.

Therefore RF1 failed the intended quality gate.

### Robustness result

The unseen-view IID-NLL penalty was much smaller after K=8 multi-view training than after fixed K=1 training:

- mod31: roughly +0.00089 versus +0.00958 nat/answer;
- GF(2): roughly +0.00046 versus +0.00944 nat/answer.

Interpretation: multi-view training showed a **view-robustness signal**, not a capacity result.

### Parallel execution

On the tested 12-layer CPU fixture:

- single view: about 28.65 ms;
- four-view Python loop: about 103.37 ms;
- four-view packed: about 72.59 ms;
- four-view packed + shared prefix: about 69.03 ms.

The shared-prefix path was about 1.50x faster than the four-view loop. This is a small CPU benchmark, not a GPU or large-model prediction.

## 8. Reachability-adjusted analytical work

The router-free analysis generalized Mirror transforms around an FFN nonlinearity:

```math
\Psi(v;sA)=e^{-sA}\phi(e^{sA}v)
```

with first-order tangent

```math
\Delta_A(v)=D_\phi(v)Av-A\phi(v).
```

The analysis established or corrected:

- a single fixed invertible view can be folded into ordinary FFN weights;
- high local gradient energy is not equivalent to task value;
- baseline reachability cost must be placed in the numerator of the efficiency ratio;
- unreachable directions must not become artificially free through a pseudoinverse;
- `K>=r+1` is only a finite-support rank bound for zero-mean codes;
- token-period zero-mean codes do not generally cancel because token sensitivities differ;
- det(Q)=1 does not guarantee good conditioning;
- data-derived dense generators have real description cost.

The canonical formulation is in [REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md](REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md).

## 9. Reachability-selected small-FFN pilot

A later exploratory implementation tested whether choosing directions with high baseline-emulation cost improves actual shared-weight continuation.

Scope:

- nonlinear 2 -> 4 -> 1 FFN;
- 17 learned shared scalars;
- 2 synthetic task families x 3 seeds;
- 5 continuation methods;
- 30 continuation comparisons total;
- learned/frozen 4x4 Mirror matrix is extra description cost;
- this was **not a Transformer capacity test**.

Result:

- reachability-selected Mirror beat same-parent Dense continuation in 2/6 conditions;
- median relative loss change versus Dense was approximately +1.15% in one family and +1.92% in the other (worse is positive here);
- local reachability score therefore did not translate reliably into improved subsequent shared-weight training.

Conclusion: **reachability is a candidate-generation diagnostic, not a sufficient selection rule.**

## 10. What is currently supported

### Fact

- deterministic multi-view execution and shared-gradient accumulation work numerically;
- reusable factors can be trained under explicit relation supervision;
- Mirror count and breadth are separate resources;
- the tested narrow gain/rotation families do not provide a stable capacity advantage;
- unseen-view robustness can improve under multi-view training;
- local reachability-only selection is insufficient.

### Interpretation

The remaining opportunity is not “add more Mirror states.” It is:

> find a small set of low-description transformations that expose useful functional directions which are expensive/interfering for ordinary updates, and validate those directions by actual shared-weight training.

### Hypothesis

Token-phase and sensor/viewpoint Mirrors may be better targets than arbitrary state IDs because they correspond to known structure in the observation process.

This hypothesis is **not yet experimentally established**.

## 11. Current proposed architecture

The current proposal is documented in [MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md](MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md).

Key points:

- one shared Transformer;
- no learned router by default;
- deterministic Mirror coordinates from sensor/token/layer/global known state;
- non-rotation Mirror family allowed;
- low-description structured generators;
- candidate selection uses reachability + task utility + retention + bytes;
- candidate adoption requires finite shared-weight training;
- sensor-parallel same-event training is the next world-model extension.

## 12. Next experimental gates

### Gate A — structured Mirror family

Compare at fixed parent and budget:

- rotation-only;
- symmetric/shear-capable structured Mirror;
- random fixed Mirror;
- Dense/dropout controls.

Require real shared-weight training gain, not only local geometry.

### Gate B — token scheduling

Compare:

- layer-only Mirror;
- token-only periodic Mirror;
- layer x token Mirror.

Audit phase shift, cache continuation, prefix-length changes, and position confounds.

### Gate C — capacity frontier

At fixed serialized bytes and controlled work, increase:

- shared-rule load;
- private-rule load;
- number of Mirror coordinates.

A capacity claim requires a larger one-view feasible region than strong Dense/regularization/MoE controls.

### Gate D — sensor-parallel world model

Use synchronized multi-sensor observations of the same latent event.

Start with separate small encoders + sensor-fixed Mirrors + one shared world-model Transformer.

Require:

- same-target performance;
- sensor swap/dropout audits;
- held-out calibration/viewpoint transfer;
- storage and compute accounting.

## 13. Claims that should not be made

Do not claim yet that:

- Mirror increases Shannon information;
- K equals effective expert count;
- K=r+1 is capacity-optimal;
- reachability eigenvalue predicts learning success;
- router-free Mirror is better than Dense;
- token periodicity increases capacity;
- sensor Mirror learns a shared physical world model;
- synthetic CPU results transfer to natural language, vision, robotics, or large-scale training.

## 14. Research decision

The project should continue, but the main question has narrowed.

Do not spend the next budget on simply increasing K.

Spend it on:

1. better structured Mirror transformations;
2. finite shared-weight validation of candidate directions;
3. token/sensor coordinates with real semantic correspondence;
4. direct capacity-frontier measurement under actual serialized-byte and compute constraints.
