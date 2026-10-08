# MA-1172 — UniSparse/Sparse Frontier coupling-aware cache allocator controls

**ADDITIONAL CONTROL ONLY.** This is not a new MA and is not a change to an earlier frozen protocol or scientific status. Current worker MA-255 and its original queue remain untouched. These stronger native controls should be introduced only through a **future preregistered protocol amendment** if that MA is activated.

## H — falsifiable extension
A cross-head coupled Mirror m allocator can improve task-relevant error under a fixed cache/code budget, but must beat native UniSparse multi-granularity selectors and query-aware native projections at identical selection masks and active operations.

## T — standalone test recipe
Create source-calibrated Q/K/V long-context bank, preserve native UniSparse composite-token graph and its query-specific sparse block selector. Measure cross-head/task Fisher or activation-second-moment weighted distortion, including off-diagonal coupling. Optimize m/head private residual allocation at whole serialized budget and compare to source-only independent head allocation, native UniSparse+one shared linear head, native KQ-SVD, packed INT4 and random equal-byte allocation. Freeze entire query-role families and masks before fresh. Use CPU exact dense audit to find false pass from independent block approximation, and one true GPU sparse-attention kernel to measure latency. Never give Mirror the full-attention target block indices while depriving native UniSparse of its selector.

**External native sources (verify exact algorithm before claiming paper reproduction):** https://proceedings.mlr.press/v306/liu26h.html; https://aclanthology.org/2026.findings-acl.1926/. Additional PA refs: PA440; PA441; PA425; PA426.

**Source/dev/fresh:** dev seeds 11/12/13; completely disjoint fresh worlds 101/102/103/104/105 ONLY if the original MA's audit is unopened; otherwise create entirely new task seeds/identities and a separate frozen amendment. Never alter a prior trial retroactively. All source adapter/selector calibration, thresholds and weights come from permitted source+dev data. Report all five fresh worlds, no audit stopping.

**Account every paid state:** copied cache, adapter library, common decoder, readout code, CPU-pinned staging, GPU high-watermark, metadata and resident folded copies. Record optimizer updates, MACs and wall P95. Compare actual output quality, not just weight reconstruction and not solely byte count.

## D — PASS / FAIL / UNCERTAIN
**PASS:** marginal m effect beats the strongest listed native alternative at native-equivalent task quality and strictly improves actual bytes or runtime with no >10% P95 regression across >=4/5 disjoint fresh worlds; strongest same-byte simple code must not explain all benefit.
**FAIL:** FAIL on false budget/quality passes, native UniSparse selected block recall/quality dominance, or any uncounted shared selector/metric bytes.
**UNCERTAIN:** absent native implementation, uncertain task labels/seed firewall, missing real serializer bytes, no real GPU offload when a GPU claim is made, or interval spanning the declared margin. Do not promote prior statuses.

## C — alternative explanation
Sparse attention selection decisions, not per-head readout coefficients, may dominate loss. The native multi-granularity index may already account for most compression while coupled allocation adds full cross-head metric storage.

## U — uncertainty and units
Model outputs and trained weights are normalized and dimensionless; m is dimensionless; quality may be nat/token or dimensionless AUROC, as defined per task. Physical storage S is nonnegative byte integer (practical unit), wall latency t in SI seconds, trained-step count as integer. Do not combine these into one opaque number. Use paired independent task/world CI and numerical replay checks; `u_c^2=u_seed^2+u_task^2+u_num^2` only under independence; add covariance terms otherwise. k_cov=2 gives an indicative expanded interval, not a 95% guarantee at n=5.

**Measurements:** attention-output relative RMSE, heldout NLL, block selection recall, physically resident KV/index bytes, calibration bytes, GPU P95. Exact original native methods must be verified from released source prior to publication-grade claims.
