# MA-573 — QuaRot Mirror coordinate sweep

PA112 and PA113 establish randomized and learned function-preserving gauges for quantization. This screen tests a one-angle hidden rotation on a two-layer model. It verifies full-precision function preservation, then measures actual packed int4 inference payload and output distortion. A direct learned-angle control uses the same selected angle, so exact alias means the result is ordinary low-dimensional gauge calibration rather than Mirror-specific functionality.


## Fresh result

The selected angle preserved full-precision function but produced mean int4 NRMSE 0.365, worse than fixed Hadamard at 0.272. Mirror payload was 997 B, 68.0% of full precision, missing the <=50% gate. Direct angle was an exact alias. **Disposition: FAIL.**
