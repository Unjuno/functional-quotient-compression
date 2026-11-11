# MA-804 — Shared Koopman operator plus sparse private nonlinear residual

Status: **FAIL for Mirror-specific value; private nonlinear capacity is necessary in the synthetic exceptional regimes**

## Prior-art boundary

PA214 motivates shared latent linear dynamics; PA18 is a reminder that compact learned scaling vectors are a strong native control for shared-basis adaptation. Here a known shared nonlinear feature is used and sparse per-regime residual amplitudes are fitted. No Deep Koopman or VeRA training is reproduced.

## Fact

Seed 80401; 8D latent state; nine regimes with two exceptional nonlinear regimes; sequence length 40; 128 train and 128 independent test sequences per regime; zero optimizer updates. Independent dynamics use 4,091 B, horizon nRMSE 1.03e-7, and max local Jacobian spectral radius .979. Shared operator without private residual uses 2,847 B; common-regime nRMSE is 1.03e-7 but exceptional-regime nRMSE rises to .1156 and overall error .05555. Shared operator plus two sparse private nonlinear amplitudes uses 3,506 B, horizon nRMSE 1.04e-7, and max local Jacobian spectral radius .979, saving 14.3% bytes versus independent. The View payload/hash exactly matches the native shared-operator plus private residual control. Active MAC proxy is 272/step versus 144 for independent (1.89×). All payloads roundtrip exactly.

## Interpretation

The result locates a private-capacity boundary: common regimes are represented by shared dynamics, while two nonlinear exceptions require private residual codes to retain long-horizon quality. The private residuals restore quality and preserve local stability in this known synthetic feature family. The full payload misses the registered 25% storage target, and the direct native control exactly aliases the Mirror encoding. The dynamic factorized-operator application also costs more MACs than independent dense transition application.

## H / T / D / C / U

- **H:** Shared low-rank Koopman dynamics plus sparse private nonlinear residuals preserve stable long-horizon behavior with at least 25% fewer complete bytes than independent regime dynamics; shared-only errors identify where private state is needed.
- **T:** Seed 80401; nine synthetic 8D regimes, two exceptional; known `tanh(wᵀz)^2` feature; 128 train/test trajectories/regime, horizon 40. Compared independent full dynamics, shared operator without residual, Mirror sparse residual, native shared/private control. Measured serialized bytes/hash, common/exceptional quality, local Jacobian spectral radius, private residual count, MAC proxy, examples, wall calibration and replay.
- **D:** **FAIL** Mirror-specific and storage gates. Shared-only misses exceptional quality; two private residuals restore it and keep rho below 1, but savings are only 14.3%, active MACs are 1.89× independent, and native shared/private control is identical.
- **C:** A single known nonlinear feature plus sparse native per-regime scaling amplitudes is enough to explain the improvement; no Mirror-specific operator is needed.
- **U:** Learned feature discovery, natural nonlinear systems, fresh regimes, longer/more diverse horizons, control metrics, gradient training and fused deployment.

Fresh remained sealed after the storage-gate miss and exact native alias. This establishes only a synthetic private-residual crossover, not a general Koopman result.
