# MA-307 — Mirror code before PackNet physical allocation

Status: SCREENING. Prior art: PA32 PackNet/Piggyback.

## H

Trying a small shared View before assigning new physical weights may delay allocation for related tasks. Novel task residuals should trigger private allocation, with earlier task functions preserved.

## T

Synthetic sparse linear tasks embedded in a D=2048 parameter space. Eight task vectors share a rank-3 basis over 128 active coordinates; four later tasks add 16-coordinate novel residuals. Compare PackNet-style independent sparse weights, Mirror plus residual allocation, generic PCA and robust shared-basis sparse fitting on the discovered support, and no-split view. Development worlds 30700–30701; fresh 30710–30712; three seeds each. Serialize indices, values, factors and metadata and count bytes.

This is a storage/allocation screen, not a trained PackNet reproduction.

## D

Pending development/fresh runs.

## C

Generic low-rank task-vector compression may delay allocation equally well; the constructed stream may make novel residual detection easy.

## U

No iterative pruning, optimizer retention, routing, natural task or neural-network accuracy evidence.

Development: Mirror splits all four novel tasks and no initial shared tasks, using ~61 residual values and 1,653 B with NRMSE ~4.1e-4. PackNet-style independent storage uses 1,536 private values and 9,358 B at exact function fit. A robust generic shared-basis fit, which estimates each code from the majority of coordinates and stores sparse outliers, uses the same 61 private values and NRMSE ~9.0e-4 at 1,952 B. This is a strong ordinary-control match; fresh evaluates frozen conditions.
