# MA-471 — ROME rank-one updates encoded by factual Mirror coordinates

Status: **PROMISING, analytic aligned-orbit screen only**  
Evidence lane: edit efficacy / specificity / storage / compute  
Branch: `research/ma-471-rome-rankone-mirror-20261009`  
Protocol frozen at `9dbb08b0`; fresh worlds 47110–47112 × seeds 0–2.

## H — Hypothesis

For factual ROME edits whose keys and target deltas lie on fixed two-dimensional rotation orbits, shared bases plus two per-fact angles will retain rank-one edit behavior and specificity with fewer actual serialized bytes than storing every ROME key/value factor. Arbitrary facts outside that orbit require private state.

## T — Test

We used an 8D analytic association with rank-one update `ΔW = v kᵀ / ||k||²`. Across 64 standalone edits per world, keys were unit vectors in a fixed 2D plane and output deltas had fixed norm 1.25 in a second 2D plane. Controls were full ROME key/value factors, generic Cartesian coefficients over the same shared planes, and Mirror angle codes. The basis tensors were included in every shared representation's payload. We measured target efficacy NRMSE, drift on random inputs orthogonal to the edited key, payload bytes at N=1/20/64, factor decode/apply MAC proxies, payload encoding wall time, and apply wall time. Fresh payloads use cloned slices so serialization cannot charge hidden parent tensor storage.

## D — PROMISING, narrowly scoped

**Facts:** Across nine fresh world/seed pairs, N=64 Mirror used 2,909B (45.45B/edit), generic coefficients used 3,421B (53.45B/edit), and full ROME factors used 5,925B (92.58B/edit). Mirror was 50.9% of ROME bytes and 85.0% of generic-coefficient bytes. Maximum target efficacy NRMSE was 1.07e-7; maximum specificity drift was 4.43e-8 for all methods. At N=20, Mirror was 2,589B vs ROME 3,109B and generic 2,717B, so it missed the frozen <=80% ROME gate at that smaller bank size. N=1 Mirror and generic payloads were both 2,461B, larger than the 1,893B ROME factors. Measured N=64 apply time was about 6.4 microseconds/edit for each method in this CPU microbenchmark; encoding times were ~2.1, 2.6 and 1.5 microseconds/edit for Mirror, generic coefficients and ROME respectively.

**Interpretation:** A fixed-norm rotation-orbit prior can amortize many rank-one ROME factors. The angle representation is smaller than Cartesian shared-plane coefficients here, but it is a direct polar reparameterization of the same two-dimensional coefficients; no new factual capacity or learned Mirror-specific function is demonstrated. The storage benefit appears only after amortization and depends on the known norm/orbit.

## C — Strongest counter-hypothesis

The gain comes from a hand-specified fixed-norm 2D manifold and ordinary polar coding. A natural ROME fact may not lie on that orbit; allowing arbitrary key/value vectors needs private components, which can erase the storage gain. The analytic setup measures no factual language efficacy or accumulated-edit interference.

## U — Unknown

Natural factual edits, learned orbit/basis discovery, off-orbit private-residual recovery, multi-edit cumulative interference, paraphrase generalization, and Transformer inference runtime are untested. The protocol-listed off-orbit private-residual control was not executed, so no result is claimed for it. MA-472 covers code composition and order effects separately.

## Decision

**FACT:** Frozen aligned-orbit fresh worlds meet N64 quality/specificity and byte gates; Mirror angle payload beats the registered generic Cartesian coefficient payload.  
**INTERPRETATION:** This is a scoped rate advantage for angle coordinates on a known fixed-norm orbit.  
**HYPOTHESIS:** Learned factual associations may have reusable low-dimensional key/value geometry, but this screen does not establish it.  
**BOUNDARY:** Analytic 8D rank-one edit, standalone per-fact evaluation; no claim of ROME reproduction on language models.
