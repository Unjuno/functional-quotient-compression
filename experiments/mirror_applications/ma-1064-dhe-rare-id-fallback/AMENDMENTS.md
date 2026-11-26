# Amendments

## AM-01 — scope rare-slice gate to items (before audit)

Development partition inspection found 0 rows with train-only user frequency <=5, versus 137 rows with train-only item frequency <=5. Therefore the original user-and-item macro rare-slice objective is not estimable for users on this split. Before any audit scoring, the registered gate was narrowed to item-frequency<=5 AUC; user slices remain reported when at least 100 development or audit rows are available. The selected cutoff remains 5 under the item-only development score. No fresh/audit labels or outcome metrics were accessed; pair IDs were hashed only to apply the registered exclusion/split and were not persisted.

## AM-02 — development result serialization fix (before audit)

DEV01 completed all development training but failed while serializing the aggregate JSON because NumPy `bool_` is not JSON serializable. The aggregate metrics were lost; no audit labels, outcomes, or metrics were accessed. The runner's JSON encoder was fixed to convert NumPy scalar values, and the identical development protocol was rerun successfully. No model architecture, data split, cutoff grid, seed, epoch candidate, or selection criterion changed.
