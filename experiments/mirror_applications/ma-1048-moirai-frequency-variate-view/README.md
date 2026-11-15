# MA-1048 — Moirai frequency × variate View

The frozen protocol uses the official Moirai-1.1-R-small checkpoint and previously unused PEMS-BAY traffic sensor data. It forms 5-minute and 15-minute views and withholds all registered frequency×sensor pairs from code fitting. See PROTOCOL.json for the fixed pair/time split, controls, and quality gates.

Development selected seasonal naive by validation CRPS; calibrated Moirai View and native-control coefficients are frozen in runs/DEV_FREEZE.json. Final-period predictions have not been generated yet.
