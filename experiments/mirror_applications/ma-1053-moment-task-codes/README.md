# MA-1053 — MOMENT task/function codes

**Status:** scoped FAIL; development and fresh audit completed.

This experiment evaluates one MOMENT-1-base encoder across ETTm2 forecasting, tiltAPB2 anomaly reconstruction and ECG5000 classification, with each task scored using its own metric.

**Mirror insertion:** this experiment adds eight paired-plane rotation coordinates per task (24 scalar codes total) to the 768-dimensional native pre-head encoder representation immediately before a shared low-rank projection, so task-specific logical functions can use different directions in one physical projection basis without storing a full projection per task.

A rotation after the shared bottleneck would be absorbable into unconstrained private task readouts, so the experiment places it before the shared projection. Controls include no-code hard sharing, equal-byte diagonal gains, independent rank-32 projections and pinned native MOMENT task interfaces.

Mirror failed the registered all-task gate: it slightly improves anomaly point F1 over diagonal gains on one event, but loses on anomaly AUPRC, forecast error and classification tolerance. The low-rank shared basis reduces adaptation state versus independent projections; this is not a Mirror-specific storage gain.

See [PROTOCOL.json](PROTOCOL.json), [RESULTS.md](RESULTS.md), [RESULTS_CORE.csv](RESULTS_CORE.csv) and [VERIFICATION.json](VERIFICATION.json).
