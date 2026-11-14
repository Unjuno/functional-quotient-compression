# MA-1003 — few-shot reactive-surface Mirror code

Status: **FAIL (development gate; fresh stage sealed)**
Branch: `research/ma-1003-reactive-surface-mirror-20261114`

## H — hypothesis

A rank-limited per-surface coordinate over one shared low-rank correction basis can adapt a frozen MACE-MP prior to reactive H₂/Cu geometries with few reference structures, preserving both energy and force quality while using at least 25% fewer actual adaptation-payload bytes than independent readouts.

## T — run and controls

**Fact.** Public NOMAD data DOI `10.17172/NOMAD/2023.05.03-2`; entries were fetched through compact partial archive queries. The frozen metadata split manifest SHA-256 is recorded in `PROTOCOL.json`. Development download-manifest SHA-256: `1413b3deaf8dfd44a5738c62ba3cba2311f31d95be939ba449e0a3edfb450177`. We used only converged, finite 56-atom Cu54H2 structures. Stage 2 supplied 46 basis-fit records across Cu100/Cu110/Cu111; stages 3 and 4 supplied 64 development records; stage 5 supplied 32 few-shot support records. Stage 1 entries in the selected sample were nonconverged and excluded. Stage 6 raw files and metrics were not opened.

The common prior was the official 32,581,838-byte MACE-MP small checkpoint (SHA-256 in `PROTOCOL.json`), CPU float32, Torch 2.14.1, MACE 0.3.14. A shared DFT energy-reference offset was fit only on stage 2 and included in payload accounting. Forces were derived from energy; NOMAD Newton labels were converted to eV/Å and cross-checked against one ASE FHI-aims output (max component difference `1.11e-16` eV/Å).

Controls: frozen MACE-MP plus stage-2 scalar calibration; rank-2 shared basis with four per-facet codes; same-rank native low-rank readout; independent full feature readouts fit from the same four-shot support. Rank 2 and 4 shots per facet were selected using stage-3/4 energy-only development grid. Both stage3→stage4 and stage4→stage3 were evaluated. No full-backbone PA301 fine-tuning was run.

## D — decision

**FAIL on the preregistered development gate.** Mirror lowers energy MAE relative to frozen calibrated MACE-MP (`0.0111–0.0150` vs `0.0874–0.0901` eV/atom) but increases force component MAE (`0.186–0.301` vs `0.082–0.090` eV/Å). Independent readout controls outperform Mirror in both energy and forces (`0.00383–0.00430` eV/atom; `0.0676–0.0803` eV/Å).

The rank-2 Mirror adaptation payload is 5,187 B versus 7,029 B independent payload (26.2% fewer bytes), meeting the adaptation-byte gate while missing quality. Including the common 32,581,838-byte checkpoint, total serialized inference state is 32,587,025 B for Mirror and 32,588,867 B for independent readouts, only 0.0057% fewer full-system bytes. Mirror and native low-rank factorization serialize to the exact same 5,187-byte payload/hash and implement the same function. The per-structure energy-readout MAC proxy is 514 for rank-2 Mirror versus 256 for independent readout. Measured CPU force-evaluation time for Mirror was 15.54/17.56 s per 32 structures versus 14.82/14.49 s for independent; one fold is 21.2% slower, outside the frozen 20% runtime gate. No Pareto adoption follows.

The failure is decisive at development, so stage-6 fresh data stayed sealed under the protocol. `RESULTS_CORE.csv` includes the initial curated-file feasibility rows, explicitly marked as exploratory/uncalibrated, and the later primary development rows.

## C — strongest counter-hypothesis

The shared basis learned from only three source facets does not span task-specific force corrections. Energy-only fitting can improve energies while producing poor gradients. An ordinary independent readout is both more accurate and has lower active readout MAC; the exact native low-rank alias shows no Mirror-specific value.

## U — remaining unknowns

- Fresh stage-6 performance (sealed after the development failure).
- Full PA301 frozen-transfer/backbone fine-tuning comparison.
- Whether force-aware adaptation or a private residual restores force quality within the byte frontier; this would require a new preregistered experiment/amendment and fresh stage IDs.
- Barrier MAE: no complete reactant→transition-state path or barrier labels were verified.
- Generalization beyond this H₂/Cu dataset.

## Fact / interpretation / hypothesis

- **Fact:** development split, selected setting, metrics, payload bytes/hashes, compute and data limitations above.
- **Interpretation:** this rank-2 energy-fitted coordinate compresses readouts but sacrifices force quality and aliases a native factorization.
- **Hypothesis:** force-aware fitting or paid private residuals might recover the quality loss, but neither has evidence in this MA.
