# MA-461 status

- Status: FAIL
- Branch: `research/ma-461-hyperformer-mirror-codes-20261008`
- Base commit: `a211269`
- Protocol frozen: yes; SHA-256 4dc721b28bb7ef1a816b796b160b6aed084ea1a1094b2c10090efb83ad704447
- Development complete: yes; seeds 46101, 46102
- Fresh/audit opened: no; sealed by frozen gate
- Results committed: 530fbdf8910e87667cfec8555d4b396492fb2c6b
- Verification record: points to result commit 530fbdf8910e87667cfec8555d4b396492fb2c6b
- Registry row: FAIL

## Decision

FAIL: Mirror uses 3,286 vs 3,854 actual bytes and 160 vs 352 MAC proxy/input, but misses the storage threshold, exceeds the quality bound on seed 46102, and exactly matches the native low-rank control on both seeds.

## Preserved implementation records

Initial delta-only evaluation metrics were superseded and preserved under `runs/superseded_delta_only_metric_*`. The common identity base cancels in RMSE, but the corrected runs and verifier explicitly apply `base + adapter`. No model setting or threshold changed; fresh data stayed sealed.
