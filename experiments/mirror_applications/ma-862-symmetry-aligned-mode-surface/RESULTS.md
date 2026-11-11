# MA-862 results

## Fact

- Four independently trained 64→32→10 MLPs were fit to identity/90°/180°/270° Digits domains for 60 epochs. Development seed 67 selected LR=.03. Fresh seeds were 271, 283, 307.
- Hidden-unit alignment used train-only activation correlation and Hungarian assignment. Permuting fc1 rows/bias and fc2 columns preserved individual endpoint logits within max absolute difference 1.91e-6.
- Aligned worst-grid accuracy was .2500, .2750, .3139; unaligned was .2306, .2833, .3139. Mean grid accuracy was .6440/.6522/.6698 aligned versus .6396/.6424/.6591 unaligned. Alignment improved mean by 0.44–1.07 pp; worst-grid effects ranged -0.83 to +1.94 pp.
- Four individual endpoint accuracies were .964–.975. The pooled standard MLP scored .8417–.8604 on the equal four-domain test mixture; interior grid views were far worse. Uniform aligned or unaligned soups were also poor (roughly .23–.32).
- Actual raw four-model payload: 43,827 B. Raw models plus charged permutation metadata: 45,799 B. Physically permuted aligned bank: 43,827 B. Twenty-five materialized aligned views: 268,599 B. Alignment metadata adds 1,972 B (4.5%) if raw checkpoints are retained and decoded.
- Mirror and native coordinates matched exactly on the grid. Alignment costs about 4.4–4.9 ms; four model training used 4,080 updates / 258,720 examples and took 1.16–1.22 sec. Alignment proxy: 17,661,952 MACs; 25-view interpolation proxy: 241,000 MACs; grid inference proxy: 21,312,000 MACs.

## Interpretation

Activation-based symmetry alignment preserves each endpoint, but it does not make their bilinear interior a useful low-loss surface in this domain-shift setup. The small average improvement does not approach the preregistered 5 pp alignment gain, and aligned grid quality remains far below the pooled standard model. The result is a negative alignment screen, not a claim that permutation symmetry is irrelevant in deeper networks.

The Mirror coordinate is only the native bilinear chart reparameterized. It adds no functions. Storing all four endpoints plus permutation metadata is larger than the raw bank, while the 25 materialized grid is larger but consists of views whose quality is poor.

## H / T / D / C / U

- **H:** Hidden permutation alignment of four task models will improve interpolation enough to form a useful compact surface.
- **T:** Four independently trained rotated-Digits MLPs; 60 epochs; dev seed 67 selected LR .03; fresh seeds 271/283/307; train-only activation-correlation Hungarian alignment; unaligned/aligned surfaces, pooled MLP, endpoint experts, soups, and materialized views compared. Actual bytes, MAC proxies and wall time measured.
- **D:** **FAIL** for the preregistered quality/alignment gate and Mirror-specific attribution. Mean grid gain is at most 1.07 pp, worst-grid gains are inconsistent, and the pooled model exceeds grid accuracy by at least 17 pp. One strict endpoint-preservation threshold (1e-6) was narrowly missed at 1.91e-6 due to float accumulation.
- **C:** The poor interior is caused mainly by task-specific function differences across rotated domains, not only hidden-unit permutation mismatch; alignment cannot make distinct decision functions convexly compatible.
- **U:** Deep-network alignment, multiple hidden layers, same-task independent basins, learned nonlinear surfaces, natural tasks, and larger datasets remain untested.
