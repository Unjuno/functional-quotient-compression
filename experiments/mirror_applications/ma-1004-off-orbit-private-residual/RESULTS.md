# MA-1004 — off-orbit private sparse residual

Status: SCREENING

## H — hypothesis

On gas-phase rMD17 tasks outside MACE-MP's crystal training distribution, a shared rank-2 molecule coordinate may miss energy or force quality; a small paid private sparse residual may recover that quality with fewer actual bytes than PA300 sparse fine-tuning.

## T — current activity

**Fact:** Pinned PA300 source at `8873c7ca05580cf7577d961ee021ef056824c54f` is MIT licensed. Four per-molecule rMD17 tensor artifacts (benzene, toluene, naphthalene, uracil) are downloaded from the public StructureCloud mirror; the parent ColabFit dataset is CC0-1.0. SHA-256s and source links are in `runs/DATA_PROVENANCE.json`. The official MACE-MP small checkpoint is the common frozen prior. Benzene schema, unit strings, and `old_indices` range were inspected; target uracil energy/force/coordinate tensors have not been parsed or scored.

The locked split uses `old_indices`: source and target support below 100,000; development 100,000–199,999; fresh 500,000–649,999. No benchmark metric has been generated.

## D — decision

**NOT ESTABLISHED.** This is experiment setup only. No adaptation model or quality control has run.

## C — strongest counter-hypothesis

A crystal-trained MACE-MP prior may be so mismatched to isolated molecules that the experiment measures domain failure rather than the private-state frontier. Any rank-2 parameter code may also exactly alias an ordinary low-rank factorization.

## U — remaining unknowns

- Available row counts in each fixed old-index window after molecule-level filtering.
- PA300 EquiSparseDeltaSTR compatibility with the pinned small MACE-MP raw model.
- Few-shot energy/force frontier, private mask size, actual payload bytes, force equivariance, and CPU runtime.
- Fresh uracil performance, which remains sealed until configuration freeze.
