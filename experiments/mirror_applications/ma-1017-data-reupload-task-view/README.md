# MA-1017 — Task-conditional data re-upload schedule

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `acce6c207` (MA-1016 evidence branch)
Prior art: PA308

## Hypothesis

H: A task phase `m_t` applied at every data re-upload block can recover four distinct logical classifiers from one shared two-qubit ansatz with fewer actual bytes than independent VQCs, while retaining accuracy at matched shots and gates.

## Mirror insertion

> **Mirror insertion:** this experiment adds task code `m_t` to both input rotation angles at every repeated data-upload interface so that one physical VQC gate skeleton can express multiple task functions without storing a separate variational circuit per task.

- Shared object: 2-qubit, 3-block data-reupload VQC ansatz and variational angles.
- View: scalar phase offset to the two qubit input rotations at every block.
- Simplest control: ordinary native task-conditioned scalar input-angle schedule.
- Strong upper control: independent per-task data offsets and variational angle lists.
- Prior art: PA308 establishes data re-uploading itself; this screen tests only the marginal task coordinate.

## Protocol

See `PROTOCOL.json` and amendment 01. A development-only selection fixed a teacher with four distinct label functions. This is an aligned synthetic teacher screen using an ideal 4-amplitude statevector and seeded shot sampling. No circuit training or fresh/audit data is used.

## Gates

- **PASS:** Mirror preserves ≥98% of independent teacher shot accuracy, saves ≥25% actual bytes versus the independent task bank, and beats the native scalar schedule control by ≥2 percentage points at equal shots/gates.
- **FAIL:** ordinary scalar schedule exactly aliases Mirror, or Mirror misses the quality/storage gate.
- **NOT ESTABLISHED:** task functions are not distinct or payload/statevector replay fails.

No QPU, hardware-noise, compiled-gate or natural-task claim is made.
