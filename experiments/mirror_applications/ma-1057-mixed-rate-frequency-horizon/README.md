# MA-1057 — Mixed-rate × horizon interpolation

This bounded screen uses one frozen official Moirai-1.1-R-small model to compare forecasts on 5/15-minute PEMS-BAY series at 1/24-hour horizons. Calibration is fit on three frequency×horizon cells and evaluated on the remaining cell. It tests whether a factorized address adds value beyond native Moirai and additive calibration; no model weights are trained.

PEMS-BAY sensors 0–13 were used in MA-1048. This experiment predeclares the next eight finite stored IDs starting at 14 and discloses reuse of the same network and calendar range. A three-point causal box mean is the recorded anti-alias aggregation for 15-minute data. Context is matched to 24 physical hours (288 or 96 samples by rate).

Required controls are native Moirai, global/frequency-only/horizon-only/additive calibration, factorized scale+offset, independent cell coefficients, and seasonal naive. Actual serialized model/checkpoint/code metadata and CPU compute are charged. See `PROTOCOL.json` and the frozen audit results after execution.
