# MRI / inverse-operator Givens View diagnostic — 2026-11-14

## Evidence

Two consecutive P0 candidates in the `MRI / inverse operators` family failed on the same scoped structural cause, on dedicated branches, with fresh metrics sealed:

- **MA-1006, VarNet:** after fair hard-tied pretraining, post-denoiser complex Givens Mirror NMSE was 0.160418, effectively tied to hard tying at 0.160539, and worse than the same-size scalar gate at 0.155246. Full inference payloads were 68,678 B for both Mirror and scalar gate. Synthetic 64×64 four-coil screen, one seed.
- **MA-1007, MoDL:** hidden-channel Givens Mirror NMSE was 0.047501, equal to hard tying at 0.047493 and worse than a three-value scalar-step control at 0.047129. Mirror payload was 6,861 B vs 6,609 B for scalar step. Synthetic 32×32 four-coil screen, one seed.

Both keep the measurement operator fixed, and both show the same failure: orthogonal Givens coordinates on a shared denoiser do not recover useful function beyond a simpler scalar modulation. One mechanism rotates the complex denoiser output; the other rotates hidden feature-channel pairs.

## Ruling

Pause further `MRI / inverse operators` candidates that reuse Givens-on-shared-denoiser views until the family is redesigned. Preserve remaining registered candidates as **UNTESTED**, not FAIL. A follow-up in this family must first propose a structurally different view aligned to measurement/coil/Fourier structure or show why the scalar-control failure mechanism no longer applies. This ruling does not cover the separately registered `MRI / multi-contrast` family (MA-1008), scanner/site adaptation methods with different insertion points, or all MRI Views.

Both experiments are synthetic mechanism screens. They do not establish performance on fastMRI acquisitions or clinical data.
