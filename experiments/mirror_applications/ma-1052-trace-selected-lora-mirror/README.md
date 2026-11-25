# MA-1052 — TRACE selected LoRA modules with task-horizon Mirror coordinates

Status: scoped FAIL; development and fresh audit completed.

This experiment asks whether a small role-conditioned rank-space coordinate adds useful adaptations on top of TRACE's own selected-LoRA and reconstructed-head method. The primary controls are native TRACE specialists, pooled hard-shared TRACE, and an equal-size diagonal rank-gain code. Generic LoRA is not used as a substitute for TRACE.

TRACE's official source implementation was not found during the initial source check. The native control will follow the paper's Gated DSIC and ProjDown descriptions, using the pinned public MOMENT-1-base checkpoint and official MOMENT model implementation. The protocol records the version discrepancy and paper ambiguity in the pruning text.

## Protocol

See [PROTOCOL.json](PROTOCOL.json) for the frozen data roles/splits, controls, storage accounting, compute metrics, and decision gates. Protocol was frozen before training, audit access, and model outputs.

## Results

See [RESULTS.md](RESULTS.md) for the H/T/D/C/U report. [RESULTS_CORE.csv](RESULTS_CORE.csv) contains macro scores and actual complete payload bytes; [RESULTS_PER_ROLE.csv](RESULTS_PER_ROLE.csv) gives role-level metrics. `runs/payload_hashes.json` preserves sizes and hashes for the inference weights (large temporary files were removed after replay verification). Canonical serialized-payload results are in `runs/dev/payload_metrics.json` and `runs/audit/payload_metrics.json`.
