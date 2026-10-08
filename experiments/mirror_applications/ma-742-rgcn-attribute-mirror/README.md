# MA-742 — Relation-attribute factorized R-GCN Mirror views

Status: **FAIL**  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Base commit: `16807a7d6dc17a834a44ed3cc793ccc28b615691`  
Experiment branch: `research/ma-742-rgcn-attribute-mirror-20261008`  
Closest prior art: PA188, R-GCN relation-basis sharing.

## H — hypothesis

A shared relation-transform basis with multiplicative Mirror coordinates composed from two semantic relation attributes can predict held-out attribute combinations using a smaller relation-code payload than free R-GCN basis coefficients, while retaining seen-relation quality.

> **Mirror insertion:** this experiment adds `m(a,b)` to the coefficient interface that selects shared R-GCN relation transforms, so unseen attribute combinations can express logical relation transforms without storing a free coefficient vector for every relation ID.

## T — what ran

- Synthetic relation-conditioned linear message operator: four values per relation attribute, 16 ordered relation pairs, with `(0,0)`, `(1,2)`, `(2,3)`, `(3,1)` withheld from fitting. Each world generates a new shared basis and rank-2 multiplicative coefficient function.
- 12 seen relations, 256 training examples each; evaluation uses 256 examples per relation (4,096 total). PyTorch 2.6.0 CPU; single thread. 600 full-batch Adam updates per model.
- Development seeds: 7421, 7422. Frozen choice: rank 2, 600 updates.
- Fresh seeds: 74201, 74202, 74203, run after the pre-fresh freeze commit `74e0d32`. No settings were changed after fresh access.
- Controls: independent full relation matrices; native R-GCN basis with free per-relation coefficients; additive relation-attribute factors; multiplicative Mirror factors.
- `native` and independent controls have no trained transform for held-out combinations; their held-out score is N/A. They are seen-quality and storage controls.
- Every training condition sees 1,843,200 example exposures over 600 updates. Training MAC values are a 3× forward-pass proxy for forward/backward work. Runtime is one full 4,096-example CPU evaluation pass per condition.

## Fresh results

| World | Mirror held-out MSE | Additive held-out MSE | Mirror / additive | Mirror seen MSE | Native basis seen MSE | Mirror / native |
|---:|---:|---:|---:|---:|---:|---:|
| 74201 | 0.478681 | 0.102453 | 4.672× | 0.000884184 | 0.000226439 | 3.905× |
| 74202 | 0.000222862 | 0.328444 | 0.000679× | 0.000227967 | 0.000229212 | 0.995× |
| 74203 | 0.000229745 | 0.289335 | 0.000794× | 0.000226174 | 0.000227142 | 0.996× |

Two of three fresh worlds pass both quality comparisons; world 74201 fails both. The preregistered gate requires all three, so overall status is **FAIL**.

## Storage, compute and runtime

- Serialized marginal relation coordinate: Mirror **104 B**, native coefficient table **264 B** (Mirror uses 39.4% of the coefficient bytes, a 60.6% reduction).
- Complete inference payload: Mirror **1,128 B**, native basis model **1,288 B** (12.4% fewer bytes). Independent full matrices use **4,104 B**.
- Mean fresh training wall: Mirror 0.704s, additive 0.693s, native basis 0.553s, independent 0.662s.
- Training MAC proxy per model: Mirror 1,592,654,400; native basis 1,592,784,000. Both use the same 600 updates and example exposures.
- Mean full-pass inference wall on CPU: Mirror 0.377ms, additive 0.379ms, native basis 0.266ms. This is one small-batch timing per condition and is not a throughput claim.
- Full packed payloads and 12 fresh model artifacts are retained under `source/payloads/`; hashes and exact bytes are checked by `source/verify_and_export.py`.

## D — decision

**FAIL:** the storage gates passed, but fresh quality was not stable across all three worlds. Mirror failed seen and held-out quality in one world despite matching both in the other two.

## C — strongest counter-hypothesis

The held-out result may depend on non-convex factor optimization and the random world/model initialization, rather than on a stable generalization property of relation-attribute Views. The single failing world is consistent with that explanation. The experiment does not separate optimization sensitivity from intrinsic ambiguity of the missing relation combinations.

## U — unconfirmed

- Whether real relation attributes in a knowledge graph provide the required factor structure.
- Whether other held-out pair layouts, optimizer initializations, or a small private residual stabilize the failing world.
- Whether this result extends beyond this synthetic linear message operator.
- Capacity near convergence; this is a fixed 600-update mechanism screen, not a capacity claim.

## Fact / interpretation / hypothesis

**Facts:** 104 B vs 264 B marginal coordinate; 1,128 B vs 1,288 B complete payload. Fresh quality gate passed 2/3 worlds and failed in world 74201. Exact serialized tensor reconstruction and all 12 payload hashes passed.

**Interpretation:** compositional multiplicative codes can recover this generator's unseen relations in some worlds, but the observed failure rate makes this setting unreliable under the preregistered criterion. Byte savings alone do not establish useful logical multiplicity.

**Hypothesis:** low-description relation Views may be useful when relation factors are learnable and well-conditioned; reliable deployment may require stronger identifiability conditions or private residual capacity.
