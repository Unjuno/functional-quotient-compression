# MA-448 — Optimizer-state Mirror superposition

## H — Hypothesis

A shared rank-4 basis and 4D code can compress task-specific Adam state while keeping five-step continuation within 1.10x exact Adam and matching PCA.

## T — Test

16D linear regression with four latent task factors; 32 tasks per seed and 3 seeds/world. Exact moments captured after 20 Adam updates; continue for five updates. Compare independent state, zero/shared state, rank-4 PCA, and learned basis plus Mirror coordinates. Invalid runs (44810–12 and 44820–22) are excluded after nonnegative-second-moment implementation corrections; compact canonical serialization A3 removes oversized tensor-view storage. Valid fresh worlds are 44830–32.

## D — FAIL

Mean continuation NRMSE: exact 0.175, shared reset 0.131, PCA 0.665, Mirror 0.822. N=32 actual bytes/task: exact 639.0B, shared 243.3B, PCA 328.9B, Mirror 328.9B. Compression reduces storage, but quality fails badly; shared reset beats both PCA and Mirror, and PCA equals Mirror bytes with better quality.

## C — Strongest counter-hypothesis

A low-rank approximation discards optimizer-state detail that affects continuation; resetting state works better in this stationary toy task.

## U — Unknown

No natural model, nonstationary task, or production checkpoint evidence.
