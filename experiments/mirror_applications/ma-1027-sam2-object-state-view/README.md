# MA-1027 — SAM2 object-state memory views

**Decision: FAIL (scoped synthetic CPU screen).**

## H — Hypothesis

Two object-specific Givens angles can recover object-specific SAM2 streaming memory from one common per-frame memory tensor, with >=25% lower full-session bytes, <=1 pp J&F loss versus native independent memory, and >=1 pp advantage over rank-1 additive.

## T — Test

We ran the official SAM2.1 Hiera-tiny video predictor on CPU at 256 px over a deterministic five-frame synthetic scene containing four moving, crossing objects. Each object received an exact mask prompt on frame 0. We compared independent memory, hard-shared mean memory, two-angle Givens Mirror memory, and rank-1 additive memory. Fitting was causal, used no future frame or ground-truth mask, and all session state was serialized.

## D — Decision and measurements

| Method | mean J&F | ID switches | object state bytes | full session bytes | FPS (single run) |
|---|---:|---:|---:|---:|---:|
| Independent | 0.8761 | 4 | 2,059,603 | 162,004,162 | 3.91 |
| Hard shared | 0.0302 | 0 | 1,568,073 | 161,512,632 | 3.81 |
| Givens Mirror | 0.0304 | 2 | 1,568,597 | 161,513,156 | 4.01 |
| Rank-1 additive | 0.3688 | 2 | 1,732,739 | 161,677,298 | 4.51 |

Mirror saves 23.84% of the object-state payload but only 0.30% of the complete session. Its J&F is 84.57 percentage points below independent memory and 33.84 points below rank-1 additive. It is effectively tied with hard sharing in quality (0.021 pp higher). The preregistered full-session storage, quality, and additive-control gates all fail.

## C — Strongest counter-hypothesis

Two rotations do not preserve the distinct spatial memory representations needed by SAM2; the Givens reconstruction diagnostic (relative RMSE 0.52895) is effectively the same as hard sharing (0.52900), while rank-1 additive is better (0.44566). The unchanged prompt/output/pointer state and model dominate total session bytes.

## U — Limits

This is one synthetic clip with four objects and one CPU model resolution. It does not establish behavior on natural video, longer occlusion, other object counts/model sizes, GPU throughput, or learned codes. FPS is one-run implementation timing. The checkpoint is external; exact hash and source provenance are recorded.

## Fact / interpretation / hypothesis

- **Fact:** deterministic forward replay reproduced all four prediction masks, payload hashes, metrics, and byte arithmetic exactly; six tests passed.
- **Interpretation:** this Mirror code does not create a useful SAM2 memory compression frontier in this setup; rank-1 additive is a stronger control.
- **Hypothesis:** increasing spatially structured private capacity or using a learned spatial transform may recover quality, but must be compared against native low-rank memory and charged in full.

Full measurements and provenance: `runs/development.json`, `runs/memory_reconstruction.json`, `VERIFICATION.json`; rerun with `source/verify_result.py` in the pinned official SAM2 checkout and checkpoint environment.
