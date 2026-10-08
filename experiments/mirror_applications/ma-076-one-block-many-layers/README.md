# MA-076 — one tied block to multiple Mirror layers

Status: PROMISING
Evidence lane: MECHANISM
Base commit: `ccf4d5c4e83992d70ccdc5db6032e428f6532380`

## Hypothesis

H: A shared physical block plus a small learned depth coordinate can recover multiple logical layer functions on aligned teachers with lower serialized payload than independent layers, while outperforming simple tied, static-LoRA, and generated-gain controls at equal update budget.

## Physical-to-logical claim

- Physical object: one 8×8 linear block reused at four depths.
- Mirror coordinate: one learned Givens angle per depth, charged in payload.
- Logical multiplicity: four depth-specific maps.
- Failure mode: arbitrary independent layer functions may require private per-layer weights; simple low-rank controls may express the teacher with similar storage and better fit.

## Prior-art delta

- PA01 and PA06 establish expert/depth sharing and generated per-step modulation as relevant existing methods.
- This screen compares ordinary tying, rank-1 static per-step LoRA, per-depth generated diagonal gain, Mirror conjugation, and an untied upper control.
- MA-247 previously failed a different fixed-budget recursive-depth screen; this candidate measures supervised per-layer contributions on aligned and independent linear teachers. It does not reuse MA-247 claims as evidence.

## Comparisons and protocol

All methods see the same four-layer input/output regression data and train for 600 Adam updates at LR 0.01. Development seeds are 76001–76002. Fresh seeds 76011–76013 are opened only if the development gates pass. Both a Givens-conjugated shared teacher and an independent-matrix teacher are evaluated. The exact protocol and gates are in `PROTOCOL.json`.

## Storage and compute

The authoritative payload is `torch.save` of the inference method metadata and learned state tensors. Training optimizer state is excluded. Every learned coordinate and modulation tensor is included. The reported MAC proxy, examples, updates, wall clock, and test throughput use the same fixed task dimensions.

## Results

The development gate passed in both aligned worlds, so the locked three fresh worlds were opened. Results are in `RESULTS_CORE.csv`; all 50 rows were replayed with exact payload byte counts.

### Fresh aligned teacher (n=3)

| Method | Median test MSE | Payload bytes | Median train wall / world |
|---|---:|---:|---:|
| tied | 0.05726 | 1,961 | 0.256 s |
| static rank-1 LoRA | 0.00945 | 2,593 | 0.323 s |
| generated gain | 0.05624 | 2,277 | 0.299 s |
| Mirror | 1.12e-13 | 2,149 | 0.966 s |
| untied | 5.28e-13 | 2,729 | 0.227 s |

Mirror passed the preregistered quality gates in 3/3 fresh worlds and used 21.3% fewer payload bytes than untied layers. It was 17.1% smaller than the byte-near static LoRA model and fit this aligned teacher more accurately. Its MAC proxy was 6.25% above the shared linear baseline from coordinate construction; measured training throughput was 0.235x untied (eager CPU implementation).

### Fresh independent-layer teacher (n=3)

Mirror test MSE was 0.684–0.762, versus 0.376–0.445 for static LoRA and near-zero for untied layers. Shared physical weights plus a one-angle-per-layer view did not recover arbitrary independent layers.

## Decision

**FACT:** The aligned Givens-conjugated teacher passed quality and byte gates in all three fresh worlds. Payload was 2,149B vs 2,729B untied. The independent teacher remained poorly fit. Mirror training throughput was about 4.25x slower than untied in this eager CPU implementation. Tests passed 3/3; 50/50 rows replayed with exact payload bytes and maximum MSE replay delta 4.68e-11.

**INTERPRETATION:** This is a storage/quality mechanism gain for a teacher that exactly matches the Mirror inductive bias. It does not improve the measured runtime frontier, and simple static low-rank control is already more accurate than tying/gain on the aligned task, though larger in serialized payload.

**HYPOTHESIS:** Depth-specific functional views are useful when layer transformations share a structured coordinate system; unrelated layer functions need private capacity.

**BOUNDARY:** Synthetic 8D linear maps, four logical depths, fixed 600-update budget, CPU only. Natural-language NLL, nonlinear Transformer blocks, near-convergence capacity, and optimized Mirror kernels are untested.
