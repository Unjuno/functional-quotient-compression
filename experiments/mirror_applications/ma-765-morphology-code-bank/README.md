# MA-765 — Compact morphology code bank with damage recovery

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

Sixteen smooth procedural morphologies were corrupted at 30% of cells and repaired for 24 steps by one shared local relaxation rule. The independent full map bank uses 59,064 B, reaches final target nRMSE `.01396`, and recovers 96.7% of initial error. Four-value morphology codes plus a deterministic generator use 1,013 B (98.3% fewer) and achieve identical error/recovery; their payload hashes exactly match the native analytic-code control. The code generator adds 196,608 simple operations, increasing the MAC proxy from 4,718,592 to 4,915,200 (4.2%). Hard tying one target reaches nRMSE `.811` and worsens the initial error.

## Interpretation

Compact analytic coordinates can encode many smooth target maps and support damage restoration under a shared local relaxation rule, with a large byte reduction and a small extra operation count. Ordinary analytic conditioning exactly explains the Mirror result. The local dynamics move toward an encoded target; this does not demonstrate self-organized NCA growth or emergent regeneration.

## Hypothesis result

**FAIL** for Mirror-specific value because the native analytic code is an exact function and payload alias. Generic code-based morphology compression passes the synthetic quality, recovery and byte gates.

## Frozen protocol / scope

PA195/PA196 motivate reuse of one update rule across target morphologies. Sixteen 32×32 maps are generated from angle, phase, frequency and bias. A fixed local recurrence relaxes damaged states toward each target while a neighborhood-average term diffuses state. Each initial map has 30% random replacement damage; 24 steps measure recovery. Compare complete serialized dense maps, hard tying, native analytic code generation and Mirror code Views. The code representation includes all 16 coordinates, IDs and grid/generator metadata. Actual bytes, normalized target error, recovery, MAC proxy and wall time are measured.

The targets are procedural smooth waves and the update rule directly relaxes toward them. This is not a learned Growing NCA, emergent morphology or biological model. No hardware runtime claim is made.

## H / T / D / C / U

- **H:** A small morphology View code bank supports many stable target behaviors and damage recovery with lower complete bytes than independent target maps.
- **T:** Seed 76501; 16 patterns; 32×32; 30% cell replacement damage; 24 recurrence steps; independent, hard-tied, native analytic code and Mirror controls.
- **D:** **FAIL** for Mirror attribution. 1,013 B vs 59,064 B with identical nRMSE `.01396` and 96.7% recovery; +4.2% operation proxy; exact native code alias.
- **C:** The procedural sine-pattern generator, rather than a learned functional view, accounts for the compression and reconstruction.
- **U:** Learned NCA morphologies, arbitrary nonprocedural shapes, self-repair after dynamics rather than target forcing, long-run attractors and robustness.
