# MA-573 — QuaRot Mirror coordinate sweep

PA112 and PA113 establish randomized and learned function-preserving gauges for quantization. This screen tests a one-angle hidden rotation on a two-layer model. It verifies full-precision function preservation, then measures actual packed int4 inference payload and output distortion. A direct learned-angle control uses the same selected angle, so exact alias means the result is ordinary low-dimensional gauge calibration rather than Mirror-specific functionality.
