# MA-333 — Mirror sign/scale orbit audit

Status: SCREENING. Prior art: PA47 monomial weight-space symmetries.

## H

Activation and normalization determine which monomial hidden-unit Views are exact gauges: positive scaling with inverse outgoing scaling should preserve ReLU, sign flips should preserve tanh, and those transformations should generally fail for GELU and normalized preactivations.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-hidden-unit signed scale coordinate `m` before the activation and compensates the outgoing weights, testing when the coordinate is a pure gauge and when it changes the function.

One fixed two-layer MLP is the shared object. The cheapest control is parameter reindexing/ordinary rescaling; independent weights are the storage reference. This is a symmetry audit, not a compression or task-quality claim.

## T

16-24-8 MLPs with ReLU, GELU, tanh, and LayerNorm+ReLU. Test hidden permutation, positive nonuniform scaling with inverse outgoing compensation, and signed flip with inverse compensation. Use development worlds 33321-33322 and fresh worlds 33331-33333; 4,096 deterministic Gaussian inputs/world. Serialize full base weights plus each view code; report output NRMSE, max absolute difference, bytes and isolated time. No training.

## Gates

PASS: permutation exact in all four conditions; positive scale exact for ReLU only; sign flip exact for tanh; nonmatching activation/normalization cases show NRMSE >1e-3. FAIL if these predictions fail. This only maps exact equivalence classes for the tested functions.

## Boundary

Fixed random synthetic networks; no learned tasks, utility, model merging or deployment. Approximate invariance and trainability are not assessed.


## Results

**H:** ReLU permits positive hidden rescaling with inverse outgoing scaling; tanh permits sign flips; permutation is exact for all tested conditions. GELU and LayerNorm+ReLU generally break the nonmatching scale/sign transforms.

**T:** Development worlds 33321–33322 and fresh worlds 33331–33333; each world tested 4 activation conditions × 4 views on 4,096 inputs. Base network fixed at 16-24-8; LayerNorm affine parameters are included and permuted. No optimizer updates. Payload bytes include model state, View code and metadata.

**D — FAIL for general sign/scale functional multiplicity:** Fresh mean output NRMSE by condition: permutation is <=1.1e-7 throughout. ReLU positive scale 6.66e-8 (exact gauge), ReLU sign flip 1.043. Tanh sign flip 0 (exact gauge), tanh positive scale .193. GELU positive scale .089 and sign flip .962. LayerNorm+ReLU positive scale .186 and sign flip 1.093. Exact gauges increase payload over baseline: permutation 83 B; positive scale 147 B; sign code 132 B.

**C:** The function-changing cases were not trained or tested for useful task behavior. Numerical behavior depends on the chosen random weights and activation implementation.

**U:** Other norms, learned task deltas, quality/utility, and inference kernels remain untested. The CPU wall-clock measurements are noisy and not used to claim a runtime gain.

### Fact / interpretation / hypothesis

**Fact:** 48 fresh rows, 12 worlds/activation/method combinations; exact payload hash/length and metric replay across all rows, max difference 0; 2 tests passed.

**Interpretation:** activation homogeneity and oddness determine whether these monomial addresses are gauge-only. Counting them as logical functions is incorrect where they are exact. Where they change functions, this screen does not show utility.

**Hypothesis:** task-specific functional views will require either a non-symmetry code with measured quality value or private residual parameters.
