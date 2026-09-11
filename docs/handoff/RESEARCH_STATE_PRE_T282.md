# Research State

## 1. Current formulation

Functional Quotient Compression (FQC) is no longer based on the hypothesis that a reversible “mirror” transform is itself a source of compression. The current formulation is:

> Identify degrees of freedom that are equivalent under the target function/task, quotient or derive those redundancies, represent the remaining shared/private structure with a low-description decoder, and optimize the actual serialized codec under a task-quality constraint.

A useful conceptual pipeline is:

`canonical / functional alignment -> quotient or hard shared structure -> private exceptions -> task-sensitive geometry -> joint codec optimization -> exact serialization -> task witness`

## 2. Established corrections

The following are treated as hard constraints on future claims:

1. **Invertible transforms are not fundamental rate reduction.** A known invertible change of variables does not by itself improve the underlying rate-distortion function when distortion is transformed consistently.
2. **Actual serialization is authoritative.** Logical bit counts can select the wrong state because headers, padding, alignment, metadata, and decoder prerequisites cost bits.
3. **Task coupling matters.** Independent per-block distortion can underestimate true distortion and can produce false hard-budget passes.
4. **Low energy is not low task value.** Spectrally tiny modes may remain decision-critical.
5. **Search work is not codec rate.** Work/query/controller savings in the synthetic scheduler line do not change codec bits unless they alter the serialized decoder DAG.
6. **Toy optimality is not real-model evidence.** Exact enumeration establishes mechanisms and algorithmic correctness only for the tested toy system.

## 3. Main research results

### 3.1 Functional / decision quotient principle

Multiple lines converge on an observational-equivalence view:

- gauge freedoms can make different parameterizations functionally equivalent;
- decoded-signature states with identical future decoder capability can be merged;
- D116 established a synthetic decision-null common-mode example in which a shared feature coefficient changes absolute prediction but leaves all pairwise action-score differences and decisions unchanged.

This motivates treating the object of compression as an equivalence class of task behavior rather than the raw parameter vector alone.

### 3.2 Serializer-aware exact optimization

The exact toy series established several codec-level mechanisms:

- **E1:** serializer overhead can change the optimum; a raw 76-bit state serialized to 88 bits and falsely passed an 80-bit logical-bit check. Exact Pareto pruning reduced 152 states to 11 without losing the serializer-aware optimum.
- **E2:** cross-block terms can invalidate diagonal distortion accounting. In the tested system, a naive model produced a false 32-bit pass whose true distortion violated the threshold.
- **E3:** tree topology is part of compression geometry, not mere metadata. For 16 leaves, about 2.09M recursive optimizer states collapsed to 65,536 decoded masks (~31.9× state reduction) under decoded-state quotienting.
- **E4–E6:** local description operations, conditional frontiers, and operation-local caches can reduce exact search substantially while preserving hard-budget correctness when their assumptions hold.
- **E7:** tree layout and precision can be complementary. In the constructed 8-leaf / 5-state example, neither tree-only nor precision-only optimization improved the incumbent, while the joint move improved distortion by 6.40% at the same serialized budget.

These are deterministic/synthetic mechanism results, not Transformer compression ratios.

### 3.3 Task-sensitive spectral geometry

The D115–D120 line produced an important representation lesson:

- D115 showed a large synthetic improvement after removing a globally harmful predictor coordinate, indicating that representation error dominated controller complexity.
- D116 converted that ablation into a reusable decision-gauge interpretation for the tested common-mode feature.
- D117 showed that the smallest tested singular mode carried only about **0.019%** of coefficient Frobenius energy but was strongly harmful to delete with hard rank reduction. Soft shrink was better.
- D118–D120 explored automatic and robust shrink rules. They improved the base rule in the synthetic environment but did not beat the best tested fixed shrink (`q=0.75`).

The durable conclusion is not the numerical q value; it is that **parameter/spectral energy is not a sufficient proxy for downstream decision value**.

### 3.4 Hard 64× certificate framework

The D50–D69 line established the accounting discipline for a future real-model claim:

- for a 16-bit scalar baseline with `N` unique paid scalars, the hard 64× target is `B <= floor(N/4)` bits;
- roots, maps, selectors, metadata, coefficients, support descriptions, and private residuals must be included unless deterministically DERIVED from already-paid acyclic decoder state;
- D56 defines a real-pilot evidence contract;
- D57 defines architecture-aware Transformer extraction constraints;
- diagnostics, sensitivity allocation, shared atoms, learned dictionaries, and root pricing are proposal mechanisms, not 64× evidence by themselves.

## 4. Synthetic scheduler line

D39–D49 and especially D70–D120 contain a substantial second research track on certified search work, caching, value-of-information scheduling, validation cost, replay, and synthetic decision controllers.

This line is retained because it can reduce the cost of future research and candidate evaluation. It is deliberately separated from codec evidence:

- `work/stream`, query count, calibration work, and controller savings are research-compute metrics;
- they must never be reported as model compression unless they change the paid serialized representation.

## 5. Reproducibility state

The handoff audit found:

- the D120 package preserves 119 original checkpoints from D1–D120; D96 is the explicit missing checkpoint;
- its outer SHA manifest and checked nested manifests are internally consistent;
- some late `*_reference.py` files are specification/pseudo-code rather than executable Python and must be reclassified during reconstruction;
- the COMPLETE handoff contains an older executable/reproducibility branch and the E1–E7 exact experiment log, but some required legacy `.npz` inputs are missing;
- some latest exact-toy results are recorded as logs without a complete canonical runner, so they must be reconstructed and marked as such.

The repository therefore treats reconstruction provenance explicitly rather than pretending all historic artifacts are immediately executable.

## 6. Current evidence boundary

| Claim | Current status |
|---|---|
| Invertible mirror alone creates fundamental compression | Rejected |
| Functional / decoded equivalence is a useful organizing principle | Supported by theory + toy/synthetic evidence |
| Serializer-aware exact optimization can change hard-budget decisions | Exact toy evidence |
| Cross-block coupling can create false passes under diagonal accounting | Exact toy evidence |
| Joint tree/precision optimization can beat coordinate optimization | Exact constructed toy evidence |
| Low spectral energy implies safe deletion | Rejected by synthetic counterexample |
| Real Transformer contains enough quotiented/shared structure for large gains | Unknown; requires real structural audit |
| Actual serialized Transformer codec beats standard baselines | Not yet demonstrated |
| 64× real-Transformer compression | **Not demonstrated** |

## 7. Immediate research transition

The project should now stop accumulating synthetic deltas as the main line and move through three concrete gates:

1. reconstruct the canonical exact optimizer and its tests;
2. run a real Transformer structural audit under the D56/D57-style evidence contract;
3. build an actual serializer and evaluate a fixed codec family with a hard lower/upper-bound feasibility test.
