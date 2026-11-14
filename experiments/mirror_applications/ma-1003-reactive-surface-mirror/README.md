# MA-1003 — Few-shot reactive-surface Mirror code

Status: SCREENING  
Evidence lane: STORAGE / RUNTIME / CAPACITY (screening only)  
Base commit: `6463b1676`  
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`  
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## Hypothesis

**H:** Given a frozen, common pretrained interatomic potential and a shared low-rank energy-readout basis learned on development surfaces, a small per-surface coordinate `m` fitted from few reference structures will reduce held-out reactive-region energy and force errors versus the frozen potential, and approach an equal-byte independent readout without storing a separate full readout per surface.

This is falsified if `m` fails to improve held-out errors, if the byte-near native low-rank control matches it (no Mirror-specific gain), or if few-shot inference requires too many private parameters to beat the independent readout payload.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to a shared low-rank correction of the frozen MACE-MP node-energy readout so that surface-specific reactive corrections can be expressed without storing an independent readout for each surface.

- Native method before adding `m`: frozen MACE-MP potential.
- Exact insertion point: additive energy correction from pooled final node features, `delta_E = pooled_node_features @ B @ m`; forces are derived by differentiating this energy with respect to coordinates.
- Persistent `m`: one fitted vector per surface/task; the shared basis `B` is also paid.
- Cheapest ordinary parameter: native low-rank linear readout / ridge adapter using the same basis and code shape.
- Physical object: common frozen MACE-MP weights.
- Claimed logical objects: task/surface-specific potentials obtained by few-shot calibration.
- Closest prior art: PA301, frozen-transfer fine-tuning of materials foundation potentials for reactions/alloys.
- Delta tested: whether low-description per-surface coordinates on one shared correction basis give a useful sample/serialized-byte frontier, beyond ordinary frozen transfer and native readout factorization.

## Data and provenance

- Public source: `wgst/ml-gas-surface`, pinned commit `72db5df1f152c86afa4de98af628b1a67268445b`, MIT-licensed repository.
- Local dataset file: `databases/h2cu/pes/h2cu_20220529_as4_forces_red_clr20240208_e0.xyz`, 4,220 extxyz frames, 25,174,977 bytes, SHA-256 `d4346a959558077447bdfefdcaaf66f761190ef29ad81898e6ff61a173b02a09`.
- NOMAD source dataset DOI `10.17172/NOMAD/2023.05.03-2`, dataset id `4r6KEYExRb-lQRUtpCfL7g`.
- Common prior: official MACE-MP small checkpoint `2023-12-10-mace-128-L0_energy_epoch-249.model`, downloaded from the ACEsuit MACE-MP release, SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`, 32,581,838 bytes. It is not committed.
- `ml-gas-surface`'s domain-trained H2/Cu checkpoint is excluded from the shared prior because it was trained on this domain dataset.
- Runtime: Python 3.12, Torch 2.14.1+cpu, MACE 0.3.14, ASE 3.29.0; CPU only.

The curated extxyz contains energies, forces, and `config_type=h2cu`, but no trajectory/path IDs; it is excluded from primary scoring. NOMAD mainfile paths expose stage, facet, and result-run IDs. The frozen primary split is: stages 1–2 shared-basis fit; stages 3–4 development selection; stage 5 few-shot target support; stage 6 whole-run-group fresh audit. Stage-6 run IDs were selected and hashed from metadata only: first 8 result-run groups per facet by the frozen SHA-256 ordering. The lock manifest is `runs/NOMAD_SPLIT_MANIFEST.json` (SHA-256 `7be703b68660dc60ea7c9e2e323d2ddc34b633943505d6dadd8e5e472f0d82c2`). Stage-6 raw files and metrics remain unopened. No barrier MAE will be claimed without a verified reactant-to-transition-state path and reference barrier.

## Controls

1. Frozen common MACE-MP prior.
2. PA301-style frozen-transfer / partial fine-tuning, if the pinned implementation can train the selected checkpoint and is reproducible in this environment.
3. Shared-basis plus task code (Mirror parameterization).
4. Same-rank native low-rank readout/code control (required attribution control; mathematically may alias Mirror exactly).
5. Independent task readout / private correction upper control where feasible.
6. Domain-trained H2/Cu MACE checkpoint is a diagnostic reference only, never a fair common-prior baseline.

## Screening observations (development feasibility only)

Three initial 56-atom structures per cell group were inspected before protocol freeze; those indices are burned and excluded from all future fresh scoring. Frozen MACE-MP force predictions ran on CPU. Per-frame time was about 0.45–1.8 s during this small diagnostic. Energy MAE was about 0.77–0.79 eV/atom on those selected structures; force vector-norm MAE ranged about 0.13–0.23 eV/Å. These numbers are a tiny feasibility diagnostic, not a held-out quality estimate or a result for Mirror adaptation. The large energy offset indicates that frozen prior calibration is essential.

## Gates

- **PROMISING for the registered application:** on a preregistered fresh set with confirmed reactive-region coverage, few-shot shared-basis adaptation improves energy and force errors over frozen prior, is within 5% of the equal-budget independent readout on both metrics, and reduces actual serialized adaptation payload by at least 25% versus independent readouts without >20% inference-compute or wall-time penalty.
- **FAIL:** frozen-transfer is not improved; Mirror fails the quality threshold; the same-rank native code/readout matches Mirror (Mirror-specific attribution FAIL); or the complete payload does not meet the byte gate.
- **NOT ESTABLISHED:** no defensible fresh path/facet split, no executable PA301 control, or the prior cannot represent the domain well enough to distinguish adaptation methods.

## Scope limits

This remains SCREENING until split groups and data lineage are validated, controls are run, actual inference payloads are serialized and measured, and fresh reactive-region scores are replayed. The few structures previously viewed are development-only. Nothing here establishes catalyst transfer generally or a reaction barrier frontier.
