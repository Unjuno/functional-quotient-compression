# MA-297 / MA-299 SETA-family diagnostic — 2026-10-08

## Scope

Two related P0 screens asked whether a compact Mirror coordinate adds useful state efficiency around SETA-style shared/private continual structures. MA-297 tested Mirror views within a discovered sparse basis. MA-299 tested whether a Mirror view should be tried before a private full matrix is allocated.

The MA-297 evidence is on `research/ma-297-seta-mirror-subspace-20261008` (report and final corrected results). MA-299 is in `experiments/mirror_applications/ma-299-split-on-share-mirror/`.

## Facts

- **MA-297:** On a fixed two-atom shared sparse basis, the Mirror code used fewer incremental bytes than a two-coefficient control, but missed its quality gate in one of three aligned fresh worlds, had 0.43× coefficient-control inference throughput, and a much larger grid-fit compute proxy. The control had the better measured quality/compute point. Unrelated functions remained expensive and required private state.
- **MA-299:** In three fresh six-task streams, Mirror split-on-share accepted three aligned and one near-aligned task, privately split one unrelated task, and used 2,337B at mean normalized MSE 0.000597. The coefficient split made the same allocation with mean MSE 0.000596 at 2,352B. Mirror saved 15B total (0.64%, below the 10% gate), with a 1.74× fit-compute proxy. Independent full matrices used 6,417B. Both shared policies delayed full allocation on this matched stream.

## Interpretation

The repeated limitation is not failure of shared/private allocation itself. It is failure to show that the Mirror coordinate adds a useful quality/byte/compute frontier over ordinary basis coefficients. The small scalar code can reduce address bytes, but the full payload saving is marginal and fitting can cost more. The native two-coefficient basis expresses the same registered Givens family directly.

MA-297's strict quality miss and poor timing, followed by MA-299's sub-1% byte edge against an equivalent native control, satisfy the stop rule for this current SETA Mirror-code family. Do not continue more candidates that reuse the same one-angle-over-fixed-shared-basis mechanism. A future retry needs a materially different insertion or task family and a new preregistered MA amendment/ID.

## Unresolved

Neither screen reproduces SETA discovery, learned routing, real online updates, or natural task streams. MA-299's zero forgetting is guaranteed by frozen stored state, not evidence of continual-learning stability. This family diagnostic does not claim that all Mirror-based allocation strategies fail.
