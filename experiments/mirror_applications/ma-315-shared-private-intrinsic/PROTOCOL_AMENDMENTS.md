# MA-315 protocol amendments before fresh access

## Amendment 1 — profile angle fitting and deterministic packed offsets

The first development-only implementation used two alternating angle/residual updates. The 25% private worlds showed that some residual tasks remained above the validation threshold even though all six residual coordinates were available. The first payload also stored a per-task 32-bit offset that is exactly reconstructible from the rank tags.

Before opening any fresh data, the solver was changed to variable projection: for each fixed candidate residual rank, project the common-plane features and targets orthogonally to that residual span, choose the support-loss-minimizing angle on the already frozen 720 point grid, then refit the residual coefficients. The payload now packs each angle followed by its residual coefficients; offsets are derived from active-rank tags and omitted as deterministically reconstructible metadata.

This is an implementation amendment based only on development behavior and storage accounting. No private fractions, seeds, dimensions, precision, thresholds, gates, or controls changed. The initial development rows and gates are retained in `protocol_variants/initial_coordinate_descent/`; all development worlds are rerun before the fresh gate is evaluated. Fresh seeds remain unopened.
