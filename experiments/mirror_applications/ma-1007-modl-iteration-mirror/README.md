# MA-1007 — MoDL tied denoiser with iteration Mirror coordinate

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Prior art: PA303, MoDL (2018).

## Hypothesis

Per-iteration Givens angles over one shared denoiser's hidden channels recover useful iteration-specific inverse priors beyond native MoDL tying and a matched per-step scalar-regularization control, while keeping the measurement operator fixed.

## Insertion

MoDL's residual CNN is shared across three unrolled steps. A single angle per step rotates adjacent pairs of the eight first-layer feature channels before the rest of the shared CNN. Mirror receives no mask, measured k-space, or CG state. The scalar control instead learns one nonnegative lambda per step in the normal-equation solve.

## Scope

The official PA303 repository uses TensorFlow 1.x and declares no root code license, so no code or checkpoint is copied. `source/modl_torch.py` implements the paper equations independently in PyTorch. The screen uses deterministic 32×32 synthetic phantoms with four analytic coil profiles on CPU. It is mechanism-only; it is not a fastMRI benchmark or clinical result.

## Controls

Independent per-step CNNs, ordinary tied MoDL, tied MoDL with scalar per-step lambda, tied MoDL with per-step Givens Mirror angles, and independent full target fine-tuning. Measurement mask and coil forward/adjoint operators are identical across methods. Fresh synthetic seeds stay sealed unless development gates pass.
