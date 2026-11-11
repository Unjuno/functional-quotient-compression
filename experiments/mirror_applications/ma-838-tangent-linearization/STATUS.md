# MA-838 status

Status: **FAIL** — scoped synthetic nonlinear/tangent screen. Fresh data remained sealed after native PCA alias and quality-gate misses.

## H / T / D / C / U

- **H:** A shared rank-4 Mirror tangent basis would preserve nonlinear heldout fine-tuning quality at >=25% fewer complete bytes than independent updates, beat random intrinsic projection, and strictly outperform native PCA.
- **T:** One-hidden-layer tanh MLP (6-12-1), seed 83801; 10 task deltas from a shared rank-4 teacher basis; six tasks fit the basis and four are adapted from support; support 256/test 256/task; delta scales .03 and .20; closed-form Jacobian least squares; zero optimizer updates. Amendment 01 raised support from 96 to 256 because the original count was below the 97 parameters. Focused test: 1 passed.
- **D:** FAIL. At scale .03, full tangent adaptation reached heldout nonlinear nRMSE .00248 and linearization residual was .00156, but rank-4 Mirror/native PCA were identical at 2,881 B and nRMSE .04838; random rank-4 reached .05463 at 2,884 B. Independent tangent payload was 4,671 B. At scale .20, linearization residual rose to .06658; full tangent nRMSE .7707, Mirror/native PCA 2.2189 at 2,888 B, random rank-4 .3369 at 2,883 B. Exact nonlinear teacher upper control was error 0. Mirror saves about 38% vs stored tangent deltas but fails the .02 quality gate and exactly aliases native PCA. Fresh remained sealed.
- **C:** The low-dimensional teacher construction favors a shared basis, yet support-estimated local tangent updates and nonlinear curvature still prevent the rank-4 code from meeting the fixed quality gate. At larger deltas, linearization itself breaks down.
- **U:** No pretrained model, actual fine-tuning optimization, real task distribution, fresh seed, or language-model quality evidence. This does not establish a general claim about LLM linearization.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror and native PCA payload hashes, bytes and predictions match at both scales. At the smaller scale, the full tangent fit passes quality while the shared rank-4 basis does not. Linearization residual increases substantially at scale .20.
- **Interpretation:** Tangent parameter storage can compress in this synthetic setup, but this particular rank-4 task basis does not preserve the chosen quality. Mirror adds no value over PCA.
- **Hypothesis:** Linearized task coordinates are useful only where the tangent residual is small and the task update family is compressible at the required heldout tolerance; a generic low-rank code is not sufficient.
