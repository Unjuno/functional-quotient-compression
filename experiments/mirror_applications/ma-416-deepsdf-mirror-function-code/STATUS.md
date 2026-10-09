# MA-416 status

Status: **FAIL at the frozen development screen**; fresh was not run.

## H
Structured geometric codes may beat unconstrained DeepSDF latent codes for translated/rotated/scaled fields.

## T
Canonical circle decoder plus five geometry codes versus jointly trained DeepSDF with five unconstrained codes; 8 training and 8 held-out ellipses; 256 support/2048 query points; seeds 41601/41602. Native affine equivalence and analytic oracle included.

## D
FAIL. Mirror query/interpolation NRMSE was 0.0847/0.0903 and 0.0931/0.1008, above 0.03/0.05 gates. DeepSDF query error was 0.1497/0.1523. Mirror total payload was 0.937x/0.934x DeepSDF, above the 0.80 limit. Mirror was 1.6–1.8x more accurate in query NRMSE at this budget, but native affine produced identical predictions. The exact analytic oracle fit the target, while both neural methods remained imperfect.

## C
The function family is aligned to explicit affine geometry, so the direct native coordinate-warp control removes any Mirror-specific claim. The frozen neural training budget may also understate converged quality.

## U
Near-convergence, broader shapes, 3D, and fresh seeds. The claim is restricted to this fixed-budget synthetic screen.

FACT: metric replay and payload replay passed after correcting interpolation metrics to use serialized FP16 state; first-pass values are retained separately. INTERPRETATION: no strict Pareto gain or Mirror-specific result. HYPOTHESIS: broader shape families may change the frontier under a new protocol.
