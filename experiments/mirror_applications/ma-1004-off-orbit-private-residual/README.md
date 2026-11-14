# MA-1004 — Off-orbit molecular task with private sparse residual

Status: SCREENING  
Evidence lane: STORAGE / RUNTIME / CAPACITY  
Base commit: `28f0dceef`  
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`  
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## Hypothesis

**H:** A low-dimensional target-molecule coordinate over one shared correction basis will be insufficient on an off-orbit molecular potential, but a small learned sparse private residual can restore both energy and force quality at less inference-state cost than the matched PA300 sparse fine-tune.

The hypothesis is falsified if Mirror-only meets the quality gate (private state is unnecessary), the private residual does not restore force quality, or a matched native sparse update is equally small/accurate (no Mirror-specific frontier).

## Mirror insertion

> **Mirror insertion:** this experiment adds molecule code `m` to a shared low-rank correction of the frozen MACE-MP interaction/readout weights so that multiple molecular potentials can be expressed without copying the full MACE-MP parameter set; a private sparse residual is charged when `m` cannot recover the target forces.

- Physical object: frozen official MACE-MP small model.
- Mirror coordinate: rank-2 code `m` over a basis of source-molecule parameter deltas.
- Private state: optional per-target sparse equivariant weight residual and its mask/index metadata.
- Logical multiplicity: molecular potential corrections for source molecules and a chemically distinct target molecule.
- Cheapest controls: native rank-2 parameter factorization; PA300 `EquiSparseDeltaSTR`; byte-matched direct sparse residual; independent full fine-tune.
- Closest prior art: PA300, official code pinned at commit `8873c7ca05580cf7577d961ee021ef056824c54f` (MIT).

## Data and provenance

- rMD17 from ColabFit Exchange/Hugging Face `colabfit/rMD17`, DOI `10.60732/682fe04a`, CC0-1.0; upstream rMD17 DOI `10.6084/m9.figshare.12672038.v3`.
- Molecules: benzene, toluene, and naphthalene as basis sources; uracil as held-out target (C/H/N/O composition, 12 atoms).
- Data carry DFT-PBE/def2-SVP energies and forces, converted from kcal/mol and kcal/mol/Å to eV and eV/Å using exact units.
- A public preprocessed mirror `StructureCloud/rMD17` provides one per-molecule tensor file; its benzene artifact SHA-256 is `dea6c7531d6e686a3c71239eca0f6d5b0c9cba721a9454dc0f2d234adcad021c`. Source is attributed to CC0 rMD17; the parent ColabFit record is authoritative for license and description.
- Prior: official MACE-MP small checkpoint SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736` (32,581,838 B). This is a crystal-trained prior applied to gas-phase molecular conformations. This is configuration/domain OOD evidence, not unseen-element evidence.

## Split lock

Use each structure's published `old_indices` only as a time-order key. No random-frame split. Freeze these absolute windows before reading labels: source tasks use `0 <= old_indices < 100000`; uracil support uses the same interval, development uses `[100000, 200000)`, and fresh audit uses `[500000, 650000)`. A fixed SHA256 order selects at most 128 source examples, 8/16/32 target support examples, up to 128 development rows, and up to 64 fresh rows.
- Deduplicate exact coordinate hashes across all partitions; remove overlapping rows from the later split without inspecting energies or forces.

The four per-molecule tensor shards were downloaded and hashed. The benzene shard was inspected for schema and `old_indices` range only; the uracil shard is present in the container, but its energies, forces, and coordinates have not been parsed or scored. Fresh uracil rows remain unselected/unopened until development config and payload are frozen.

## Controls

1. Frozen MACE-MP with per-task energy zero-point calibration learned on allowed support frames.
2. Shared rank-2 Mirror basis + target code `m`.
3. Native rank-2 parameter factorization with the same basis and code (exact attribution control).
4. Official PA300 `EquiSparseDeltaSTR` update, mask and threshold state charged in payload bytes.
5. Mirror + private sparse residual, with mask and indices charged.
6. Full fine-tuning as an upper control when CPU resources allow.

All controls use the same target examples, units, source prior, and held-out structures. Forces are derivatives of total energy; force equivariance is checked.

## Gates

- **PROMISING:** Mirror+private residual meets the frozen quality tolerance against PA300 sparse fine-tune for both energy and force, reduces actual total adaptation payload by at least 25%, and adds no more than 20% measured inference time; Mirror must also outperform its exact native factorization alias to support a Mirror-specific claim.
- **FAIL:** Mirror-only fails and private residual still misses quality, the residual erases the storage savings, or native/PA300 controls match the result at lower bytes or compute.
- **NOT ESTABLISHED:** no valid per-molecule split, unavailable raw-data provenance, or inability to reproduce the sparse control.

## Boundary

This tests gas-phase rMD17 molecular conformations against a crystal-trained prior. It does not establish bulk materials performance, unseen-element transfer, reaction barriers, or chemistry-wide generalization. The shared basis/code hypothesis is evaluated only for the listed molecule set.
