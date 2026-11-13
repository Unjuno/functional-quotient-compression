# MA-1001 — Mirror force-code stability across long rollouts

Status: **PROMISING for fixed-code synthetic NVE stability only**. MatterSim/MACE dynamics remain **NOT ESTABLISHED**.
Branch: `research/ma-1001-md-view-stability-20261113`

## H / T / D / C / U

**H — Hypothesis:** a fixed material Mirror code in an energy-derived force law remains stable over long NVE integration and matches its native coefficient alias; a separately perturbed force field is detectable as unstable/nonconservative.

**T — Test:** Synthetic 4-particle Lennard-Jones model, velocity Verlet, dt=.002, 3,000 NVE steps, two epsilon scale codes, fresh seeds 100111/100113/100117. Compared Mirror energy code, native per-task coefficients, independent per-task parameters, and a constant force-head perturbation negative control. No training updates. Actual serialized payloads include all parameters/code/metadata. Per-world batch times 3.87s, 2.30s, 1.81s are joint runner times; no per-method latency claim.

**D — Decision:** **PROMISING** for the scoped stability mechanism. All six Mirror trajectories remained finite and had maximum relative total-energy drift 5.48e-09; Mirror and native trajectories had identical final positions (max absolute difference 0). The force-offset negative control reached relative drift 0.005494 and center-of-mass displacement 0.04124. However, Mirror exactly aliases native coefficients, has 794 B payload vs 552 B independent (+43.8%), and 146 vs 144 MAC proxy/step. No storage or compute frontier win.

**C — Strongest counter-hypothesis:** NVE stability is explained by the same fixed conservative Lennard-Jones potential and integrator. Mirror contributes no unique dynamics; all methods use the same coefficient values and force law.

**U — Unconfirmed:** real learned potentials, periodic materials, NVT/NPT sampling, phase stability, long-horizon OOD trajectories, and real inference throughput.

## Facts

- Fresh seeds: 100111, 100113, 100117; 3,000 steps each; all Mirror/native/independent trajectories remained finite.
- Worst Mirror drift: 5.48e-09; worst negative-control drift: 0.005494; maximum Mirror/native final-position difference: 0.
- Mirror serialized payload: 794 B; native/independent: 552 B. Mirror MAC proxy/step is 146 vs 144.
- Serialized states and drift replay passed in all fresh worlds.

## Interpretation

A fixed code in a conservative energy potential did not destabilize these trajectories. That is useful as a physical-validity boundary, but it is not Mirror-specific and does not transfer to learned material models.

## Hypothesis for follow-up

MA-1003 few-shot reactive-surface adaptation is the next eligible P0 in this lane. It needs a real reaction dataset or an explicitly scoped synthetic boundary, sparse fine-tuning controls, and reaction-path/force metrics.
