# MA-520 status

- Status: SCREENING
- Branch: `research/ma-520-function-vector-distillation-20261009`
- Protocol SHA-256 recorded in freeze.json; fixed before code implementation and development.
- Development seeds 52001/52002: not run.
- Fresh seeds 52011–52013: locked, unopened.
- Registry/status board: screening after protocol freeze.

## Next action

Implement the frozen learned rank-four decoder and controls, run development only, then apply the registered gates before deciding whether fresh seeds can open.

## H / T / D / C / U

- **H:** compact FV teacher distillation can preserve causal held-out task effect and improve by >=0.10 gold-logprob nats over native PCA at <=0.50x explicit FV bytes.
- **T:** pinned Pythia-70m, 16 function vectors, 12 decoder-fit tasks, four held-out tasks, two development seeds, 2,000 updates, four registered controls. No development run yet.
- **D:** screening; no evidence evaluated.
- **C:** standard PCA may match or exceed a learned low-rank linear decoder.
- **U:** all quality, storage, compute, and alias outcomes remain unknown.
