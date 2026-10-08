# MA-971 — LightPro physical coupler bank with sparse Mirror task code

Status: **NOT ESTABLISHED — hardware blocker before experiment**
Branch: `research/ma-971-lightpro-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Selection draw: 9; see `source/random_draw.json` and the complete frozen pool `source/selection_pool.csv`.

## H — falsifiable hypothesis

A small task Mirror code can represent useful task-specific optical operators with fewer programmed photonic devices/bits and lower reconfiguration energy/time than fully programmed LightPro couplers, while preserving optical task quality under real calibrated loss and thermal drift.

## Prior-art delta

PA286 reports phase-change tunable directional couplers, programmable photonic matrix-vector multiplication and architecture search/pruning. The Mirror-specific claim concerns reduced physical settings, switching time and total energy under real optical loss and drift.

## T — execution and blocker

No model training or experiment was run. Environment inspection found no physical photonic device, no LightPro/Luceda/Tidy3D/Meep package, and only CPU PyTorch. The registration requires physical coupler programming and calibrated optical loss, thermal drift, switch time and energy; this environment cannot judge those quantities. Probe details are in `source/hardware_probe.json`.

## D — NOT ESTABLISHED

No result was produced. This is a hardware availability blocker, not a negative scientific result. No synthetic photonic simulation was substituted for physical measurements. Fresh/audit data were not opened.

## C — strongest counter-hypothesis

Any apparent savings in an ideal optical matrix simulation could disappear after calibration, insertion loss, thermal control, switching and controller costs are charged; native full programmability and pruning may already be sufficient.

## U — still unknown

Useful accuracy, configured device count, bits per task, calibrated loss/crosstalk, drift tolerance, switch energy, settling time and the actual storage/runtime Pareto frontier all remain unknown.
