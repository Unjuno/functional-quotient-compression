# MA-1040 status

- Status: FAIL
- Branch: `research/ma-1040-scann-metric-mirror-20261115`
- Base commit: `1e4cfb9fa` (MA-1039 integration state)
- Protocol and source freezes: see `ARTIFACT_PROVENANCE.json`; audit qrels opened only after development lock `358a31b90`
- Data/model: official ArgUana archive and pinned Nomic Embed v1.5 INT8 ONNX; CPU, 256-token amended screen
- Index: ScaNN 1.4.2 AVQ, one shared 8,674-document index
- Development: complete; selected short +0.30 rad, long +0.15 rad
- Fresh audit: 705 disjoint qids; quality, payload and five-repeat latency evaluated
- Results: exact ranking/metric/payload replay; 3 tests pass
- Registry row, claim ledger, status board and worker queue: updated; integration commit pending

## Decision

The predeclared quality gate fails. Mirror nDCG gain is +0.005 pp and Recall@10 gain +0.142 pp, both under +0.5 pp; MRR falls 0.031 pp. A matched native linear query transform gives exactly the same rankings. Mirror adds 95 B to the 169,461,177 B native deployment. Repeated p95 latency is 0.4496 ms vs 0.4413 ms native.

## Next action

Run final registry, replay and test checks; push this branch; then advance to P0 MA-1041 (DiskANN graph traversal with query-state View).
