# MA-740 — Few-observation operator adaptation by Mirror code only

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

The frozen shared model alone has mean test nRMSE `.678`. After the development-only ridge amendment, the four-phase Mirror adaptation gives `.2246/.0143/.00443` nRMSE at 4/8/16 observations per task. The regularized eight-value CAVIA context gives `.6819/.2247/.0120`. Mirror whole-bank payload is 1,100–1,101 B versus 1,150–1,157 B for context: about 4–5% savings. Its task-state portion is 693–694 B versus 952–959 B, a ratio around .73, above the frozen .65 limit. Mirror spends 988/1,176/1,968 forward evaluations versus context's 64/128/256, respectively. Native phase coordinates have exact hash, bytes, outputs and compute equality with Mirror. Results are in `RESULTS_CORE.csv`; the initial unregularized control run is preserved separately.

## Interpretation

The structured phase code adapts this analytic function family from 8 or 16 observations and can beat a generic eight-value coefficient context in held-out error. At 4 observations it misses the absolute quality gate. Its smaller task coordinate yields only a small whole-payload reduction, and nonlinear phase fitting takes 7.7–15.4 times as many forward evaluations. A native phase coordinate is the same parameterization, so this does not show Mirror-specific value.

## Hypothesis result

**FAIL.** The all-observation quality gate fails at n=4, the storage ratio misses .65, the adaptation compute ratio exceeds 2x, and direct native phase coordinates alias exactly. The three fresh seeds remain sealed.

## Frozen protocol and amendment

PA187 DeepONet and PA129 CAVIA. The shared analytic operator is `sum_k A_k cos(kx+phase_k)` for four fixed frequencies. Sixteen task phase vectors are adapted from 4/8/16 noisy observations and evaluated on 256 disjoint inputs/task. Controls are frozen shared weights, native four-phase coordinates, eight-value linear context, and full eight-coefficient fitting. Actual serialized model-plus-task bytes, task-only bytes, function evaluations and operation proxies are recorded. No shared model weights update.

Amendment A01 changes the generic context control from unstable unregularized least squares to ridge lambda `1e-4`, selected from a development-only sweep before any fresh seed was opened. `RESULTS_INITIAL_UNREGULARIZED.csv` preserves the initial output, including its n=8 instability. Fresh seeds 74002–74004 are not opened because development misses the preregistered gate.

This is an analytic Fourier mechanism screen, not a neural network, DeepONet or physical PDE result.

## H / T / D / C / U

- **H:** Four phase coordinates recover new operators from sparse observations with comparable quality, less state and acceptable adaptation compute versus a generic context.
- **T:** Dev seed 74001; 16 tasks; 4/8/16 calibration points; four frequencies; 256 held-out inputs/task; phase least-squares, regularized CAVIA-style coefficient context, native phase code and frozen base.
- **D:** **FAIL.** n=4 quality misses; whole-bank savings are only 4–5%; task-state ratio is .73; nonlinear adaptation costs 7.7–15.4x; native phase control aliases exactly.
- **C:** Function-family alignment favors phase coordinates, while generic contexts have more degrees of freedom; a direct native phase code captures the same gain without Mirror attribution.
- **U:** Fresh tasks, learned neural operator, physical regimes, alternate phase optimizer and practical sensor noise distributions.
