# MA-526 — SAE feature atoms for function-vector interventions

Status: **FAIL** (frozen development screen; fresh seeds sealed). Branch: `research/ma-526-sae-feature-mirror-atoms-20261009`.
Prior art: PA102, *SAEs Are Good for Steering — If You Select the Right Features*.

## H — Hypothesis

A shared pool of 64 pretrained SAE atoms with eight signed coefficients per function would preserve held-out FV causal effect, use at most half the explicit-FV bytes, and beat native SAE top-eight activations and global OMP-8 by at least 0.10 gold-logprob nats in both seeds.

## T — Execution

We used pinned Pythia-70m and a pretrained tied SAE from `Elriggs/pythia-70M-deduped-sae`, revision `36c2027509efad69836aa4231999c9b717848ecd`, artifact `pythia-70m-deduped_r4_gpt_neox.layers.3.pt`, SHA-256 `85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4` (4,204,391 B). It has 2,048 tied atoms over the 512-dimensional layer-3 residual stream. We reused MA-516 task/split/FV extraction: 12 tasks selected the pool, four were held out, and development seeds were 52601/52602. The Mirror code used eight signed pursuit coefficients restricted to the shared pool. Controls were no intervention, explicit FVs, native SAE top-eight activations, and global-dictionary OMP-8. Fresh seeds 52611–52613 remain unopened.

The incremental Mirror payload was 3,582 B versus 35,070 B explicit FVs, but global OMP-8 used 3,316 B. Standalone Pythia+SAE+Mirror deployment was 172,352,597 B, which is 4,172,903 B larger than Pythia plus explicit FVs at 168,179,694 B. The selected pools shared 57 of 64 IDs across seeds. See RESULTS_CORE.csv for causal scores, reconstruction quality, actual bytes and compute.

## D — Decision

**FAIL** under the frozen gates. Mirror reconstruction had relative RMSE .799/.781 and held-out gold-logprob losses of 1.463/1.567 nats versus explicit FVs. Accuracy was .1875/.1875 versus .25/.1875 explicit, missing the .05 tolerance on the first seed; the .10-nat criterion also fails. The Mirror pool beat global OMP gold log probability by .187/.048 nats, missing the required .10 margin on seed two, and used more incremental bytes (3,582 vs 3,316 B). SAE top-eight activation was much worse than explicit FVs. Although the code bank is small when an SAE is already resident, standalone deployment is larger than the explicit-FV system. Fresh data stayed sealed.

## C — Strongest counter-hypothesis

The pretrained SAE features are not aligned with these support-derived function-vector directions. Restricting to a selected pool discards useful residual directions. Ordinary global sparse pursuit explains the small coefficient representation, with slightly smaller code bytes. The SAE's 4.2 MB artifact overwhelms any full-system savings.

## U — Boundaries

This is one pretrained 4x SAE on one 70M model and four held-out relation tasks. It does not establish performance on open-ended generation, other SAE dictionaries or nonlinear feature transforms.

## Evidence classification

- **Facts:** two fixed development seeds; three tests pass; all eight inference payload hashes, metrics, task splits and pool-selection audit replay exactly. The initial safe-loader preflight failure is preserved and Amendment 1 changed only the loader shell before metrics were generated. Fresh seeds were not accessed.
- **Interpretation:** SAE atoms form a reproducible selected pool, but these sparse codes miss FV task behavior and do not improve the native sparse-coding storage frontier. Full deployment bytes increase because the SAE dictionary is an additional required asset.
- **Hypothesis:** a different feature selection rule or feature-space transform could preserve more task effect, but that requires the separately registered MA-527 mechanism.
