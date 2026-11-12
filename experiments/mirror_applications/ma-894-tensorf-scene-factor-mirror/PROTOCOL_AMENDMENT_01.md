# MA-894 protocol amendment 01

Date: 2026-11-12, after the preregistered fresh metrics were opened.

This amendment records two measurement-scope issues found during the post-run audit. It does not change seeds, fitted parameters, quality values, or the FAIL decision.

1. **Inference payload scope:** The frozen protocol's `paid_objects` list said “development and fresh scene codes.” The serializer measured the actual inference state needed to reconstruct the scored fresh scenes: shared factors plus the three fresh-scene codes and metadata. Development-scene coefficients are training-only and are not needed to reconstruct the scored fields. They were therefore excluded from the inference payload bytes. The reported byte values are deployment inference payloads; they do not include optimizer/training state.
2. **Query wall-clock:** The independent-dense timing path wrapped and reshaped a zero-copy NumPy array without explicitly gathering queried coordinates. Therefore the query wall-clock figures are not comparable and are excluded from the compute decision. Quality, payload bytes, fit/code-fit timings, MAC proxies, and the native-CP alias are unchanged. Query-runtime frontier is **NOT ESTABLISHED**.

No fresh-seed hyperparameter or method change was made. The original locked protocol and raw `RESULTS_CORE.csv` are preserved; this amendment narrows interpretation and documents the accounting/timing boundary.
