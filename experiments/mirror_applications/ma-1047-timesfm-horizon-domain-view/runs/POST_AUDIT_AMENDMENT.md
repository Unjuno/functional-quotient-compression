# MA-1047 post-audit reporting amendments

The final-20% TimesFM predictions and targets were generated only after runs/DEV_FREEZE.json was saved. The frozen candidate selected on validation was native TimesFM with identity output mapping.

AM-01 repairs a reducer omission: the frozen protocol listed seasonal naive, but the first audit reducer did not score it. The added metric uses saved audit origins and historical values plus the already saved targets. It does not refit any method or change Mirror, native calibration, forecast, or threshold values.

AM-02 records the frozen additive output table as a canonical ten-float representation. The serialized parameters reproduce the existing fitted additive 2×4 table exactly. Mirror coordinates, fitted tables, model outputs, and metrics remain unchanged. This is only a payload-accounting correction.

The current audit metric file was regenerated from the exact saved forecasts and targets. A second deterministic replay produced the same SHA-256: 35c86035387f4bbfd0db2f3d5899b4680ea878350a93eb9fdd4758b861ce1bf2.

The report-only scoring source changed after the development freeze to include the missing preregistered control and compact additive serialization. The original frozen code hashes remain in DEV_FREEZE.json; final source hashes and the amendments are recorded in VERIFICATION.json and PROTOCOL.json.
