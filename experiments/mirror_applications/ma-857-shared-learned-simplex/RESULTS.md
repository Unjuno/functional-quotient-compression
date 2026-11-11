# MA-857 results

## Fact

- Three jointly trained MLP vertices (64→32→10) were optimized for 80 epochs using randomly sampled uniform barycentric coordinates. Each training batch used domain rotations sampled according to the same coordinate. Development seed 41 selected LR=.1. Fresh initialization/order seeds: 131, 149, 167; fixed stratified Digits split, domains identity/90°/180° rotation.
- The ten-coordinate grid worst accuracy was .8861, .8722, .8944; paired pooled standard MLP accuracy was .9417, .9407, .9463. The worst-grid deficits are 5.56, 6.85, and 5.19 percentage points, so the preregistered 2 pp quality gate fails on all fresh seeds. The three line-trained vertices themselves scored .9333–.9556.
- Independently trained domain-specific vertex models scored .9722–.9833 per domain. Their three-model payload was 33,087 B, slightly below the simplex payload of 33,215 B. A single pooled standard model was 11,933 B. The 10 materialized simplex view states were 107,563 B, 3.24× the endpoint-plus-code payload.
- Mirror decode versus native barycentric interpolation had maximum absolute logit differences from 1.91e-6 to 2.86e-6, within the amended float32 tolerance 5e-6. Reloaded payload grid accuracies matched exactly. Per-run metrics are in fresh_results.json; RESULTS_CORE.csv includes serialized bytes, updates/examples, compute proxies, and controls.
- Training took 1.08–1.12 seconds per simplex fresh run and 1.18–1.39 seconds for the three independent domain models combined. Ten-view test query proxy: 8,524,800 MACs; coordinate decode proxy: 72,300 MACs.

## Interpretation

Jointly trained simplex views can be represented much more compactly than ten separately materialized checkpoints. This does not improve the quality/storage frontier against relevant controls: the independent three-model bank is marginally smaller and substantially more accurate at vertices, while one pooled model is much smaller and more accurate than the worst simplex view. Interior mixture quality remains weak despite training on sampled mixture coordinates.

The Mirror chart only re-encodes ordinary barycentric weights. Float32 numerical tolerance confirms equivalent outputs; it adds no functional freedom beyond native interpolation. The ten grid points are sampled views, not ten independent capacities.

## Hypothesis

Joint barycentric training can create smooth low-loss domain-mixture classifiers when the endpoint/domain correspondence is well matched; this small rotation-domain screen does not meet that quality condition.

## H / T / D / C / U

- **H:** Jointly trained vertices can support useful intermediate classifiers for matching three-domain data mixtures, with a compact barycentric view code.
- **T:** 64→32→10 MLPs; 80 epochs; Digits identity/90°/180° rotations; development seed 41 selected LR .1; fresh seeds 131/149/167; pooled MLP, three independent domain experts, model soup, and materialized-grid controls. Actual payload, update/example counts, compute proxies, and wall time measured.
- **D:** **FAIL** for the preregistered quality/storage frontier and Mirror-specific attribution. Grid quality missed the standard pooled MLP by at least 5.19 pp; the independent bank is 128 B smaller; Mirror/native interpolation differ only by float32 roundoff.
- **C:** Convex parameter interpolation cannot represent robust functions for all rotated domains; low loss on training samples does not imply good interior-domain calibration, while an ordinary pooled model is a stronger compact control.
- **U:** Larger architectures, learned code distributions, near-convergence or fixed-byte frontier, unseen rotations/tasks, and optimized quantized payloads remain untested.
