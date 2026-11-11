# MA-856 results

## Fact

- Protocol: PA231-style line training on sklearn Digits, 64→32→10 MLP, 100 epochs, fixed stratified 60/20/20 split. Development seed 31 selected learning rate 0.1 and cosine penalty β=0.1 by mean validation NLL over 17 α points. Fresh initialization/order seeds 61, 79, 97 used those locked settings. β=1 (PA231 default) and β=0 were controls.
- The heldout partition is shared across replicates. These are training seed replicates, not unseen-task or unseen-dataset tests.
- On the selected β=.1 line, worst-grid accuracy was .9611, .9667, .9611; corresponding standard MLP accuracies were .9833, .9806, .9806. The preregistered 2 percentage-point quality gate missed on seed 61 (2.22 points) and passed on seeds 79/97 (1.39/1.94 points). Endpoint disagreement was .0194–.0250.
- The two-endpoint plus 17-code payload was 22,665 bytes in each replicate, versus 182,743 bytes for 17 materialized line models (8.06× smaller). The independent two-model bank was 22,477 bytes.
- Native α interpolation and Mirror m=2α−1 produced exactly equal logits (max difference 0). Reloaded endpoint/code payloads and materialized models also had zero max logit difference.
- β=1 worst-grid accuracy ranged .9556–.9639; β=0 ranged .9639–.9722. The ordinary single model was more accurate than selected line worst-grid performance on all replicates. Fresh selected-line training took 1.10–1.25 sec; β=1 took 1.19–1.34 sec and β=0 1.19–1.34 sec on this CPU run. Full per-seed rows and active compute proxies are in `RESULTS_CORE.csv`; raw replay data is `fresh_results.json`.

## Interpretation

A single learned line stores 17 sampled functions much more compactly than 17 independent serialized checkpoints, but the independent two-model bank is slightly smaller than the endpoint-plus-code payload and has higher mean accuracy in these runs. The quality gate is not consistent across fresh seeds. This supports only a narrow storage result for materialized views, not a general storage/quality frontier improvement.

The Mirror coordinate is a reparameterization of ordinary linear interpolation and has no distinct functional or compute behavior. It does not establish Mirror-specific benefit or added independent capacity. α multiplicity counts views, not independent functions.

## Hypothesis

A learned low-loss line can provide useful intermediate predictors on some small classification tasks; this result is sensitive to training seed and does not establish transfer to larger models, language modeling, or unseen tasks.

## H / T / D / C / U

- **H:** PA231 line training can yield multiple useful diverse neural predictors from two endpoint tensors, and small Mirror coordinates can address them compactly.
- **T:** 64→32→10 MLP on fixed Digits split; 100 epochs; development seed 31 selected LR .1/β .1; fresh seeds 61/79/97; β=1 and β=0 controls; standard MLP and independent two-model storage/quality controls; actual serialized payload and compute proxies measured.
- **D:** **FAIL** for Mirror-specific attribution and for the preregistered all-seed quality gate. Narrowly, endpoint storage beats 17 materialized copies, but not the two-model bank.
- **C:** The result is ordinary interpolation between trained endpoint weights; α already provides the entire coordinate, and the saved bytes comparison against 17 checkpoints is not the relevant strongest quality-matched control.
- **U:** Other tasks/architectures, unseen task generalization, near-convergence/fixed-byte quality frontier, optimized storage formats, and inference latency under batching remain untested.
