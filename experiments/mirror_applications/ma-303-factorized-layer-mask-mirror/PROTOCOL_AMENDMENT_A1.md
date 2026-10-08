# MA-303 protocol amendment A1

This amendment is committed before any fresh seed is run. It corrects an implementation gap found after development: the initial runner generated support/validation/test arrays but only scored test arrays and did not execute the declared validation-selected private fallback.

The original numerical gates, methods, seeds, teacher construction, byte thresholds, and task split are unchanged. For direct-factorized and Mirror methods, the runner now measures validation nMSE on the preregistered 128 validation vectors for each task-layer pair; if it exceeds 0.05, that pair's teacher mask is stored as a charged private binary mask. Support examples are recorded as available examples but do not update parameters because this is a post-fit synthetic representation screen with oracle planted factor codes. Fresh seeds remain 30311, 30312, and 30313 and have not been accessed as of this amendment.

Development results from the earlier runner remain development-only. The corrected runner is frozen and will be rerun on both development seeds before fresh evaluation.
