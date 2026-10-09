# MA-311 status

- Status: FAIL for practical Mirror-specific gain
- Branch: `research/ma-311-intrinsic-mirror-task-code-20261009`
- Frozen protocol: `c272f746`
- Fresh: complete, 3 worlds × 3 seeds × 2 strata; 54 result rows
- Verification: complete

## H
A shared intrinsic vector plus per-task Mirror angle can compress task coordinates inside a random intrinsic subspace without losing query quality.

## T
D=128, intrinsic d=16, eight tasks; direct least-squares intrinsic controls, jointly trained Mirror (800 updates), generic PCA. Fresh worlds 31110–31112 × seeds0–2; both Givens-aligned and independent task strata. Actual payload includes U and task representations.

## D
FAIL for practical gain. On aligned tasks Mirror is 4,314 B / NRMSE 1.97e-5; PCA is 4,424 B / 2.90e-7; direct intrinsic is 4,510 B / 2.80e-7. The 81% smaller task-code state becomes only 4.3% total-payload saving after charging U. Mirror costs ~0.915 s and 800 updates; direct fitting is ~0.00047 s. On independent tasks, Mirror NRMSE ~0.843 while direct stays ~2.85e-7.

## C
More tasks could amortize U and analytic or faster angle fitting could reduce compute; neither was tested.

## U
No pretrained model, real task, scaling to more tasks, efficient inference kernel, or alternative optimizer evidence. Independent tasks demonstrate private-coordinate need.
