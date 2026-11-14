# MA-1006 — VarNet cascade phase-view reconstruction screen

Status: **SCREENING — amendment 01 rerun pending**  
Branch: `research/ma-1006-varnet-mirror-20261114`  
Prior art: PA302; official fastMRI source commit `91f2df4711adbb6d643df1810f234e4abcf5881b` (MIT).

## H — hypothesis

A per-cascade complex Givens angle on one shared VarNet denoiser recovers scanner-shift reconstruction quality lost by hard tying, at lower serialized state than independent cascade weights, and beats an equally small scalar-gate control while leaving the VarNet data-consistency operator unchanged.

## T — current activity

**Fact:** Official `fastmri.models.varnet.VarNet` is installed from the pinned MIT repository source. Environment: CPU only, no acquired fastMRI dataset. The bounded protocol uses synthetic 64×64 analytic phantoms and smooth four-coil sensitivity profiles; it is not a fastMRI or clinical evaluation.

**Fact:** The first development run is retained at `runs/DEV_LOCK_pre_amendment01.json` and `runs/DEV_MODELS_pre_amendment01.pt`. It is excluded from the terminal result because it formed the hard-tied baseline by averaging independently trained cascade denoisers. Amendment 01 changes the control to train the denoiser tied from initialization on the same Scanner-A examples as the independent model. The Scanner-B support and development seeds remain unchanged; fresh seeds have not been generated or evaluated.

## D — decision

**NOT ESTABLISHED.** Amendment 01 fair-control development rerun is pending. No fresh evaluation.

## C — strongest counter-hypothesis

Even under the earlier control, complex phase rotation did not beat a scalar gate. A better tied pretraining initialization could help Mirror indirectly, but it may not change the result.

## U — remaining unknowns

- The amended tied-pretraining comparison on Scanner-B development cases.
- Whether the Givens view beats same-count scalar gates after equal tied pretraining.
- Fresh synthetic-world replay if and only if the amended development gate passes.
- Natural fastMRI behavior and full-scale runtime, outside this synthetic CPU screen.
