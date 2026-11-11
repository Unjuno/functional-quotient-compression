# MA-858 results

## Fact

- A 3×3 tensor-product quadratic Bezier surface of 64→32→10 MLP weights was trained for 60 epochs on four rotated Digits domains. Development seed 47 selected LR=.1. Fresh training seeds were 181, 199, 211 on one fixed stratified source-image split.
- Fresh worst-grid accuracy over 25 coordinates was .7361, .7333, .7389. The same-seed pooled standard MLP scored .9062, .8868, .9194. All three runs miss the 2 percentage-point quality gate by a wide margin. Surface corner accuracies ranged .8750–.9028.
- Four independent domain models scored about .97–.98 per domain; their actual serialized bank was 43,827 B. The nine-control-point surface payload was 97,399 B. One pooled MLP was 11,933 B. The 25 materialized views cost 268,599 B (2.76× the surface payload). Native four-corner bilinear interpolation had worst-grid accuracy .1556–.2917.
- Native Bezier coordinates and Mirror coordinates m=2(u,v)-1 produced exactly equal logits on the tested float32 grid. Reloaded control-point and materialized payloads also matched outputs exactly (max logit difference 0).
- Per fresh run, surface training used 1,020 updates and 64,680 examples in 1.28–1.37 seconds. The 25-view query proxy is 21,312,000 MACs plus 542,250 control-blend MACs. Complete metrics and controls are in RESULTS_CORE.csv and fresh_results.json.

## Interpretation

The learned surface compresses 25 materialized grid models into nine control points, but its output quality is far below a single pooled model and four independent experts. It also costs 2.22× the independent four-model payload. Ordinary bilinear interpolation of independent corners performs even worse on interior views, so it does not explain away the poor learned surface; however, this does not produce a useful quality/storage frontier.

The Mirror coordinate is only a reparameterization of the native Bezier coordinates. There is no Mirror-specific functional gain. The 25 grid points are sampled views, not 25 independent functions or capacity units.

## Hypothesis

A jointly learned low-dimensional surface can connect useful model functions across two task dimensions when the training objective preserves low loss throughout the chart. This small rotation-domain setup fails to produce such a low-loss surface.

## H / T / D / C / U

- **H:** A jointly trained quadratic Bezier chart can provide accurate classifiers over four-domain mixtures with fewer bytes than materialized view checkpoints.
- **T:** Nine trainable control MLPs; 60 epochs; Digits identity/90°/180°/270° domains; development seed 47 selected LR .1; fresh seeds 181/199/211; pooled model, four independent domain models, model soup, native bilinear corner interpolation, and materialized 25-view controls. Actual serialized bytes, updates/examples, MAC proxies, and wall time measured.
- **D:** **FAIL** for the preregistered quality frontier and Mirror-specific attribution. Worst-grid accuracy is over 16 pp below the pooled model; the control surface is larger than and less accurate than the independent bank; Mirror and native coordinates are identical.
- **C:** Convex interpolation of independently or jointly learned weights is poorly aligned with the multimodal rotated-domain decision boundaries; the low-dimensional surface cannot preserve the separate domain solutions throughout the interior.
- **U:** Larger surfaces, nonconvex/low-rank control-point parameterization, unseen rotations/tasks, convergence-quality frontier, and actual optimized inference kernels remain untested.
