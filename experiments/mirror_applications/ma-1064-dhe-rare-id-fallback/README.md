# MA-1064 — DHE rare-ID private fallback

This experiment extends the table-free DHE baseline from MA-1063 with per-ID state allocated only to IDs meeting train-only frequency cutoffs. It compares ordinary residuals and replacement vectors with Givens Mirror-only and Mirror-plus-residual codes. User and item identities selected for the inference payload, all coordinate/residual values, and their metadata count toward actual serialized bytes.

The MA-1063 audit partition is excluded from MA-1064 train, development, and audit data because its labels were already scored. A new deterministic split is applied to the remaining ratings. Data access terms are in `DATASET_TERMS.md`; no raw ratings or per-ID payload metadata will be committed.

The DHE base implementation is reused from MA-1063 and pinned in `FRESH_FREEZE.json`. The DHE decoder and GMF head are frozen while per-ID state is trained, isolating the incremental value of the private fallback. This is a scoped k=128 CPU screen, not an official DHE reproduction.
