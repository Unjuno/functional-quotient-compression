# MA-932 results

## H — hypothesis

Exact agreement on a shared task output does not establish that source and target representations contain the same information. A bridge can preserve the common task while losing a target-private feature and its counterfactual response.

## T — trial

Five fresh analytic worlds (seeds 93211, 93217, 93223, 93229, 93231), 4,096 examples each (2,048 fit / 2,048 held out), CPU. A common threshold feature drives the shared task; target has a second independent private threshold feature; source has an independent structured Fourier nuisance in that coordinate. Compared native target, a fixed rank-1 common-coordinate stitch, an unrestricted train-split affine map, and Fourier structured-noise representation. No optimizer updates.

## D — FAIL (scoped validity claim)

### Fact

- Shared-task output maximum absolute error for the minimal stitch: **0.0e+00** across all five seeds. The unrestricted affine control also matched target task outputs to a maximum 6.7e-16.
- Held-out private-feature AUC: native **1.000**, minimal stitch **0.500**, full affine **0.500**, structured Fourier nuisance **0.505**.
- Counterfactual private-coordinate intervention mean response: native **1.0**, stitch **0.0**.
- Mean actual serialized fixture bytes (includes paid deterministic feature coefficients, task/probe heads, bridge and metadata): native 1511 B, stitch 1489 B, full affine 1247 B, Fourier control 693 B. This tiny constructed fixture is not a model storage comparison; affine was smaller than the stitch serialization. Reload replay max absolute error 0.0e+00.
- Each seed used 4096 examples, 0 updates, 8192 operation proxy, and mean elapsed wall time 0.0297s.

### Interpretation

Output agreement on one task does not imply informational equivalence. The private feature is deliberately outside the shared task, so the result identifies the limit of the inference, not a failure of the shared-task stitch itself. The unrestricted affine map and structured nuisance confirm that a more flexible connector or structured activations do not conjure an independent feature.

### Hypothesis

A valid stitching claim needs task-relevant counterfactual and feature-preservation audits in addition to output matching. This toy is a falsification example only; natural models may share more information.

## C — strongest counter-hypothesis

The private bit was constructed independent of the common task and deliberately omitted from the source. Any connector must fail to transport it. The test demonstrates logical insufficiency of output-only evidence, not the empirical frequency of this failure in pretrained models.

## U — unconfirmed

No trained LMs, PA241 datasets, causal feature interventions in natural representations, or semantic tasks were used. The synthetic byte sizes are not a deployment Pareto result.
