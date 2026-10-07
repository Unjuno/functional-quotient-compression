# MA-249 status

- Status: **PROMISING (aligned synthetic head-sharing mechanism)**
- Branch: `research/ma-249-future-head-views-20261007`
- Base commit: `4fe9b5fe51bfcd488ddff9a9702f8cb9bee4b08a`
- Development: complete; LR 0.01 selected on world 24900
- Fresh/audit: complete; worlds 24901–24903
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- **H:** A single shared output head with offset-specific Givens views can recover multiple MTP heads when they share the matching view family, while using fewer serialized bytes.
- **T:** Six head methods; 16D frozen features; four categorical future offsets; 12 outputs; aligned and independent-head teacher modes; 1,200 AdamW updates; dev world 24900; fresh worlds 24901–24903; common LR 0.01.
- **D:** PROMISING on the aligned mechanism: Mirror reached 100% top-1 agreement in 3/3 fresh worlds with 4,384B vs MTP 6,497B (-32.5%). Independent matrices defeated Mirror; rank-1 private residual improved but did not close the gap. Current CPU throughput was 0.42x MTP.
- **C:** The aligned teacher was intentionally constructed from Givens views, and the current CPU implementation is not optimized.
- **U:** Natural language, accepted-token rate, longer horizons, trainable trunk, fixed-byte near-convergence capacity, optimized kernels, and minimum private rank.

## Verification

Four tests passed. Fresh replay covered all 30 method/world/mode rows; max KL delta 4.92e-11, top-1 delta 4.38e-9, payload bytes exact 30/30.
