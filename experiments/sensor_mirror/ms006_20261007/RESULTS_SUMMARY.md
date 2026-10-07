# MS006: Backpropagation vs Gaussian ES × Mirror

Date: 2026-10-07 JST. Evidence: small synthetic differentiable MLP optimization study; no Transformer/GPU/real-sensor/capacity claim.

Using the same MS005 parents and 16 fixed shear views, reverse-mode Backprop beat the corresponding Gaussian ES method in all 3 fresh world/init pairs under both 120 update/generation opportunities and a measured 2.0 s optimizer-loop budget.

Fixed 120 median full-16-view changed-law NMSE: BP identity 0.005413, BP 16-Mirror mean 0.003675, ES identity 0.010657, ES 16-Mirror 0.009963. Paired BP-mirror vs ES-mirror reduction: 59.94%, 73.09%, 64.06%.

Equal 2 s, dev-selected cosine schedules: BP identity 0.000502, BP 16-Mirror mean 0.002119, ES identity 0.009457, ES 16-Mirror 0.010188. Paired identity-BP vs identity-ES reduction: 93.37-96.05%.

MS005 exact-gradient diagnostic for K16/128-branch independent ES had median cosine 0.2065 to exact autograd gradient. The current decision is therefore: use Backprop for differentiable shared world-core weights; retain ES for low-dimensional/discrete Mirror/role/sensor-code or black-box outer optimization.

Audit: 24 primary model rows reloaded and recomputed with maximum metric difference 0.0; 4 unit tests pass. Historical branches and main are untouched.
