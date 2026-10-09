# Excluded MA-393 implementation probe

The first development invocation used a separate random projection for each latent width. Consequently, the nominal +1-coordinate adaptive control did not contain the original k-dimensional projection subspace, so it did not implement the preregistered “add one ordinary coordinate” comparator. The two raw JSON files and eight NPZ payloads are retained here for provenance only and are excluded from all MA-393 metrics and gates.

Correctness fix: generate one deterministic nested projection per frequency band and use its first k or k+1 columns. Fix committed as `d07256bd`; four tests pass. The corrected two development seeds were rerun before any fresh seed was opened.
