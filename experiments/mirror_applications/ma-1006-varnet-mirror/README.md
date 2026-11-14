# MA-1006 — VarNet cascade phase-view reconstruction screen

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `05f2e4c72`  
Closest prior art: PA302, official fastMRI VarNet. MIT source is pinned in `source/FASTMRI_SOURCE.json`.

## Hypothesis

A shared cascade denoiser plus per-cascade complex Givens angles recovers scanner-shift reconstruction quality lost by hard tying, at lower serialized state than independent cascade weights, and beats an equally small scalar-gate control.

## Mirror insertion

The view rotates the real/imaginary two-vector of the shared denoiser output after the U-Net and before the unmodified VarNet sensitivity expansion and learned soft-DC subtraction. The view receives neither the sampling mask nor acquired k-space. Three view angles are the only Mirror coordinates. The scalar-gate control uses three real output scales.

## Scope and prior art

This is a CPU, small-model mechanism screen on deterministic analytic multi-coil phantoms. The fastMRI VarNet code is from pinned MIT commit `91f2df4711adbb6d643df1810f234e4abcf5881b`; it is not the large pretrained fastMRI challenge model. No clinical claim is made.

## Data and split

Scanner A coil maps define pretraining examples. Scanner B changes smooth coil amplitude and phase profiles. Target support, development and three fresh seed sets are deterministically generated from disjoint seeds. Fresh metrics remain sealed until model and payload are frozen. The complete fixed protocol is in `PROTOCOL.json`.

## Controls

1. Independent-cascade VarNet fine-tuned on target support.
2. One shared cascade denoiser, hard tied.
3. Shared denoiser with three scalar gates.
4. Shared denoiser with three Givens angles (Mirror).

The sensitivity estimator and native cascade DC scalars are frozen for view-only methods. Report acquired-line consistency error because the reference VarNet uses soft data consistency, not exact projection.
