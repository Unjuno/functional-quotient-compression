# MA-1049 — PatchTST variate-specific Mirror View

Status: SCREENING
Evidence lane: MECHANISM
Base commit: `430f93c1a`
Prior art: PA329

## Hypothesis

H: A two-scalar per-sensor rotation-scale View on the first two dimensions of the patch embedding lets a shared channel-independent PatchTST recover sensor-specific behavior on late unseen windows, beating both hard sharing and a byte-matched diagonal FiLM control.

## Mirror insertion

- Native method: PatchTST channel-independent shared patch encoder.
- Insertion point: output of the patch input embedding, before positional encoding.
- Persistent `m`: per-sensor angle and isotropic gain.
- Cheapest equivalent control: two per-sensor diagonal FiLM scales.

## Comparisons

Hard shared model; rotation-scale Mirror; matched diagonal FiLM; optional independent linear forecast head. All details, seeds, split, and gates are frozen in `PROTOCOL.json` before opening the dataset.

## Storage and compute

Actual serialized inference payload bytes include model, view/control state and metadata. Record examples, updates, wall-clock and compute proxy. Fixed-budget screening is not a capacity claim.
