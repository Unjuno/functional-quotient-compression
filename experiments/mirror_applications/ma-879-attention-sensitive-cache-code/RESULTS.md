# MA-879 results — FAIL

## H — hypothesis

With source support fixed to heads `[0,4,7]`, optimizing per-role rank-32 shared-basis KV codes against the target causal next-token objective would improve held-out NLL over unmodified PCA codes, remain within 0.50 nat of target-native and full weighted-ridge controls, and use no more than half the full-map payload.

## T — treatment and controls

Pythia-70M-deduped → SmolLM2-135M, Wikitext-2. Development used seed 829 on train (24 calibration / 16 held-out prompts), three learning rates and 24 updates each; frozen choice: Adam LR 0.002. Fresh runs used validation calibration and test prompts, seeds 983/997/1009, 12 calibration and 16 test prompts per seed, and 24 updates per seed. Model weights were frozen. Controls: native target prefill, full attention-weighted affine ridge maps, and rank-32 PCA codes without target-loss updates. CPU only; no GPU latency claim.

Validation calibration and test IDs were disjoint within each seed. Across seeds, all 36 calibration texts and all 48 test texts were unique, with no text overlap between splits. The post-hoc metric-only replay retained seeds, prompts, initialization, update count and frozen LR. NLLs and gate results exactly match the initial fresh output in `fresh_screening_run.json`; logit KL/RMSE were added without changing treatment or decision gates.

## D — decision: FAIL

| Seed | Native NLL | Full ridge | Rank-32 PCA | Tuned rank-32 | Gain vs PCA | Delta vs native | Gap vs full | KL (nat) | Logit RMSE | Payload (B) | Saved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 983 | 3.085 | 5.341 | 12.994 | 10.935 | 2.059 | 7.849 | 5.594 | 8.187 | 10.993 | 573,125 / 2,996,165 | 80.87% |
| 997 | 1.911 | 4.617 | 10.425 | 10.369 | 0.057 | 8.457 | 5.752 | 8.581 | 10.580 | 573,125 / 2,996,165 | 80.87% |
| 1009 | 2.850 | 4.471 | 8.372 | 8.809 | -0.437 | 5.959 | 4.338 | 6.963 | 10.765 | 573,125 / 2,996,165 | 80.87% |

Mean tuned NLL was 10.038 nat, versus PCA 10.597, full ridge 4.810, native 2.615. Tuning improved over PCA by 0.560 nat on average, but missed the ≥0.10 nat gate in two seeds. Every seed missed both +0.50 nat quality limits: tuned-vs-native gaps were 5.96–8.46 nat, tuned-vs-full gaps 4.34–5.75 nat. The byte gate passed: 573,125 B vs 2,996,165 B (80.87% saved; 19.13% of full-map bytes).

### Fact / Interpretation / Hypothesis

- **Fact:** target-loss updates lowered mean held-out NLL by 0.560 nat from projected PCA, yet stayed 5.96–8.46 nat above native and 4.34–5.75 nat above full maps. Tuning did not change payload bytes. Tuned native-to-bridge logit KL was 6.96–8.58 nat.
- **Interpretation:** this is a modest fixed-budget learning-efficiency signal for two seeds, not useful bridge quality or capacity. The storage reduction is on a poor-quality point and does not improve the quality/storage frontier.
- **Hypothesis:** rank-32 affine maps and fixed sparse source support omit information needed to approximate target KV behavior; 24 one-token updates cannot repair that structural mismatch. More calibration, a richer basis, or private residuals may help but add compute/state and need a new experiment.

## C — strongest counter-hypothesis

The ordinary shared-basis optimizer is a weak low-rank control at this update budget; longer optimization, more contexts, or a richer basis could close some gap. This does not establish that every cross-model KV translator or attention-sensitive objective fails.

## U — unknowns

Only one ordered pair, Wikitext-2, one-token probes, fixed source support, rank 32 and CPU were tested. Multi-token trajectories, longer contexts, other pairs/architectures, larger bases, private residuals, serving latency and GPU kernels remain untested. Direct low-rank coefficient optimization aliases the proposed View coordinates; there is no Mirror-specific evidence.

## Compute and payload

Fresh wall time per seed: 26.3s, 38.9s, 37.4s; mean 34.2s/world. The 24-update code fit took 5.7s, 9.9s, 10.2ss. Mean CPU tuned-map application time was 29.6 ms/test prompt. Actual serialized payload includes mean/basis, all role codes and references, pair identifier and header; common model weights are reported separately.
