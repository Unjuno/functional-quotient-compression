# MA-796 status

- Status: FAIL for Mirror-specific attribution; scoped synthetic recurrent-composition screen passes its independent comparison.
- Branch: `research/ma-796-lowrank-rnn-task-views-20261111`
- Base commit: `1101144d`
- Development: seed 79601; two heldout diagonal task compositions.
- Fresh/audit: sealed after exact native low-rank alias.
- Verification: one focused test passes; all payload reconstructions are exact; metric replay checked.

## H / T / D / C / U

- **H:** Recurrent Views recover unseen compositions with stable rollout, causal component relevance and lower bytes than independent recurrent matrices.
- **T:** 8D synthetic linear RNN, 9 two-factor tasks, 2 held out, rank 2, horizon 32; full-matrix, shared-only, View and native low-rank controls.
- **D:** **FAIL** Mirror attribution: exact native low-rank alias. View is 28.8% smaller than independent with near-zero heldout error and both components pass ablation, within the synthetic scope.
- **C:** This is exactly the shared additive low-rank recurrent model described by native conditional RNN parameterization.
- **U:** Natural tasks, training, nonlinear RNNs, task inference, fresh seeds and deployment runtime.

## Next action

Record the scoped FAIL in the registry and proceed to MA-797.
