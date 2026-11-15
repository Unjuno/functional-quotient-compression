# MA-1046 status

- Status: FAIL
- Branch: `research/ma-1046-chronos-distribution-mirror-20261115`
- Official Chronos-T5-Tiny revision and input provenance are in `runs/INPUT_MANIFEST.json`.
- Frozen code selected on ETTh1/ETTm1; fresh audit completed on Weather and Australian Electricity using seeds 104604–104606.
- Normalized CRPS fails the registered gate against zero-shot and native temperature on both datasets. Normalized-Hermite control aliases Mirror within floating-point error.
- Complete payload and measured metrics are recorded in `RESULTS.md`, `RESULTS_CORE.csv`, and `runs/audit/AUDIT_METRICS.json`.
- Prior invalid Weather attempt and reducer corrections remain disclosed in protocol amendment history; only final fresh seeds support this decision.
