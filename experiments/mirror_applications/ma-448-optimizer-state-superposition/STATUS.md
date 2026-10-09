# MA-448 status

**FAIL — Mirror state compression loses continuation quality and is matched by PCA.**

## H / T

Tested whether a shared rank-4 basis plus per-task Mirror code can compress Adam moment state while preserving five-step continuation on 16D tasks. Two initial fresh batches were invalidated because reconstructed second moments could be negative; a third compact-serialization correction removed oversized tensor backing storage. Only worlds 44830–44832 are valid.

## D — Fact

Across valid worlds, continuation NRMSE: exact Adam 0.175, shared reset 0.131, PCA 0.665, Mirror 0.822. Actual compact N=32 payload bytes/task: exact 639.0B, shared reset 243.3B, PCA 328.9B, Mirror 328.9B. Mirror state reconstruction NRMSE was 0.853; PCA 0.853. Median continuation wall was approximately 0.94ms for Mirror vs 0.95ms exact, with 256 reconstruction MAC/task.

## Interpretation

The low-rank code reduces bytes versus exact moments, but quality collapses and shared reset is both smaller and better on this proxy. PCA has the same serialized size and better quality than Mirror; no Mirror-specific value is established.

## C — Strongest counter-hypothesis

The task states are not accurately rank-4 compressible for optimizer continuation, and carrying Adam history is not useful here; zeroing state is a stronger, cheaper baseline.

## U — Unknown

Natural model optimizer states, larger models, stochastic nonstationary tasks, and quality under genuine task switching remain untested.
