# MA-520 status

- Status: **FAIL**
- Branch: `research/ma-520-function-vector-distillation-20261009`
- Protocol fixed and committed before implementation or development.
- Development seeds 52001/52002: complete; Amendment 1 corrected the explicit-FP32 optimizer/update counter and preserved the initial output; canonical replay verified exact.
- Fresh seeds 52011–52013: sealed and unopened.
- Registry/status board: FAIL after reconciliation.

## H / T / D / C / U

- **H:** a rank-four learned FV decoder would preserve held-out causal quality within 0.10 nat and 0.05 accuracy of explicit FVs, fit within half the explicit bytes, and beat native PCA by >=0.10 nat in both seeds.
- **T:** pinned Pythia-70m; 16 FVs; 12 decoder-fit and four held-out tasks; seeds 52001/52002; exactly 2,000 updates; explicit FP32/int8, PCA and no-intervention controls.
- **D:** FAIL. Learned decoder loses 0.607/0.859 nats versus explicit and differs from native PCA by only +0.000159/−0.000128 nats. Payload is 12,692 B; int8 FVs are 10,188 B and remain within 0.0017 nat of explicit.
- **C:** this is a standard low-rank shared basis; PCA matches it, and int8 is a stronger storage/quality control.
- **U:** other task families, model sizes, nonlinear decoders, and broad language behavior remain untested.

## Evidence classification

- **Facts:** see RESULTS_CORE.csv and VERIFICATION.json; fresh remained sealed.
- **Interpretation:** compression relative to FP32 storage did not deliver the registered causal quality or Mirror-attribution improvement.
- **Hypothesis:** task-aware training may change the tradeoff, but this screen did not test it.
