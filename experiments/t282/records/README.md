# Historical result witnesses

These records describe previous runs, not a fresh evaluation in the packaging audit.
Historical relative/absolute paths in JSON refer to their original source packages;
they are not necessarily executable paths in this curated repository layout.

- `T271_summary.json` is a selected-field summary, retaining the original source hash.
- `T280_long_context_checks.json` has compact JSON formatting; all values are preserved.
- CSV line endings were normalized to LF without changing field values or row order.
- The remaining copied JSON witnesses retain their original run scope and flags.
  In particular, `github_write_performed: false` describes the historical experiment,
  not the later repository handoff commit.
- The 44 prompt texts and complete legacy runners are not included in this curated lane.
  The original code-results ZIP is SHA-indexed in `provenance/t282/SOURCE_ARCHIVES.json`.
  Do not claim this lane alone reruns every historical quality evaluation.
- The eight calibration probes and three frozen artifact locks are sufficient for the
  separate CPU reconstruction entry point, provided the exact trusted checkpoint exists.

Do not alter these records to reflect new Mac/MPS measurements. Save a new named run instead.
