# MA-1038 status

- Status: **FAIL** for the registered Mirror-specific improvement hypothesis.
- Branch: `research/ma-1038-colbertv2-centroid-role-view-20261115`
- Base commit: `8e73da3cbb0386bd4405dd8488cda2e511709eb4`
- Protocol frozen: `4ae21b523c310c4bdc6b60d7234b2e2f6b8725fd`
- Initial runner frozen before audit: `bf7f7052d9ce67d83d7ff9e05bd58c814ca8a85d`
- Fresh/audit opened: yes, BEIR NFCorpus official test qrels, after protocol and implementation freeze.
- Development choices: +0.20/+0.20 rad; short/long threshold 22 tokens; selected only on SciFact train qrels.
- Results committed: pending final integration commit.
- Verification: PASS; exact metric replay (max difference 0) and exact payload file-size replay.
- Tests: 3 passed.
- Runtime: NOT ESTABLISHED; repeated CPU search passes ranged 23.1–79.9 s in order-sensitive executions.

## Next action

Register this scoped FAIL and advance to P0 MA-1039 PLAID centroid pruning with a task View.

## Blockers

None for this screen. PLAID production traversal and independent query encoders were outside this bounded ColBERTv2 full-MaxSim experiment.

## Decisions / rulings

- Native query-side inverse Givens is exactly the same score function as rotating decoded document vectors. The native control matches Mirror quality and bytes; no Mirror-specific claim is supported.
- A first run stopped after audit scoring at result serialization due to a variable-name error. The frozen development choice was recorded, the results were replayed without retuning, and an independent verification replay matched all metrics exactly.
