# MA-833 — KARC basis with system-specific dynamical readouts

## H / T / D / C / U

**H:** A shared explicit Fourier basis plus compact system Views preserves heldout dynamical forecasts with lower full bytes than independent closed-form readouts and improves on native low-rank KARC readout factorization.

**T:** Synthetic 4D nonlinear map using Fourier features of current state; eight stable systems; rank-2 readout perturbations around a shared center; 2,048 train and 4,096 test initial states; 8-step rollouts; seed 83301; no optimizer updates. Compared independent readouts, shared readout only, native low-rank readout, and Mirror readout Views.

**D: FAIL for Mirror-specific attribution.** Mirror uses 6,259 B and reaches heldout rollout nRMSE 3.19e-8, versus 8,420 B for independent readouts. Native low-rank readout uses 6,267 B with the same error, same finite rollouts and same 2,097,152 feature evaluations. The shared readout alone uses 1,693 B but misses quality (.0355). The Mirror/native 8 B difference is NPZ layout.

**C:** Standard low-rank KARC readout factorization accounts for the complete result. The shared/private frontier is visible, but not Mirror-specific.

**U:** Actual KARC model, closed-form fitting from data, natural system forecasting, unseen-system inference, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Complete payload bytes, rollout errors, feature evaluations, finite status, hashes and reconstruction errors are in `RESULTS_CORE.csv`.
- **Interpretation:** A low-rank private readout restores system variation, but native low-rank KARC matches it.
- **Hypothesis:** Mirror value may require an online system-code inference or retention mechanism beyond ordinary low-rank readout storage.
