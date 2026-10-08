# MA-366 accidentally exposed fresh IDs

The runner was invoked without `--dev-only` after development results were available, so it generated rows for seeds 36611–36613 before the development gate was adjudicated. Those rows are preserved here for provenance only. They are excluded from `RESULTS_CORE.csv`, the decision, and every fresh/generalization claim. Fresh split integrity is invalid; do not treat these as confirmatory results or use them for tuning.

The raw complete runner output remains at `../../artifacts/results.csv`. Only seeds 36601 and 36602 are used as development evidence.
