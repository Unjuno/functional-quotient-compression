# MA-1039 status

- Status: FAIL
- Branch: `research/ma-1039-plaid-mirror-pruning-20261115`
- Base commit: `720481fcc9fda56eb63483d3d42c7e000d8688f5`
- Source/protocol freeze: `b03b1137f62be54aa24836091900b535e06b9194`
- Result status: audit completed; independent metric/payload/data-map replay exact
- Development complete: yes
- Fresh/audit opened: yes, after the frozen development angle file
- Results committed: yes (`facae087ea605d29b8a76bd1c4073b1ec4bb8736`)
- Verification committed: yes (`facae087ea605d29b8a76bd1c4073b1ec4bb8736`)
- Registry row updated: yes (final integration commit pending)

## Next action

MA-1039 is closed. Registry, claim ledger, status board and worker queue now record the scoped FAIL. Run final registry integrity and push the research branch; then begin P0 MA-1040 on its dedicated branch.

## Blockers

None. Runtime is NOT ESTABLISHED because query timings were collected once per arm/query without repeated isolated calibration.

## Decisions / rulings

- All arms use the same official ColBERTv2 1-bit residual index with 512 centroid partitions, `ncells=2`, centroid threshold 0.45 and `ndocs=4096`.
- Before audit metrics were inspected, the codebook training sample was fixed at 4,096 deterministic passages to contain CPU build time. All 18,049 candidate documents remain in the index. This CPU-screen deviation is included in `PROTOCOL.json`.
- The official CPU batch size is 64; batch-size microbenchmarks yielded bit-identical token embeddings. Timing changes are not a quality claim.
- Native query-side inverse Givens reproduces centroid-bank View scores within 3.88e-7, candidate sets exactly, and top-100 rankings exactly on all audit queries; treat as ordinary routing freedom.
- Two report-only recovery replays were required after complete evaluation because archive hashing and frozen development summary fields were initially referenced incorrectly. Their audit outputs were not shown or saved; no tuning used audit metrics.
- The 20-query audit uses a bounded collection selected by all positive qrels plus 512 deterministic background documents, not the full 171k-document benchmark.
