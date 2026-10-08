# MA-401 status

- Status: **FAIL** (digits context-conditioning screen)
- Branch: `research/ma-401-film-mirror-feature-conditioning-20261008`
- Protocol frozen before development: yes; selected LR 0.01 on worlds 40100/40101
- Fresh worlds 40110/40111/40112 evaluated once at frozen settings
- Results: 168 rows across methods, contexts, worlds, and development LRs
- Verification: 3 tests pass; payload hashes, exact serialization, and metrics replay checked
- Next candidate: MA-403 (token-wise generated Mirror modulation)

## H — hypothesis

Four-angle context-specific Givens features improve four-context digit classification over a shared MLP and beat same-dimensional FiLM at better bytes/compute.

## T — execution

Digits v1.8.0 with fixed hash; four deterministic pixel permutations; same stratified 60/20/20 base identities reused across contexts. Shared 64-64-10 MLP, 600 updates, 100 context-code updates. Development worlds 40100/40101 selected LR 0.01; fresh worlds 40110/40111/40112. Controls: no code, 4-angle Mirror, 4-value grouped FiLM, 8-value grouped affine FiLM, rank-1 output adapter, independent MLP per context.

## D — decision

**FAIL.** Fresh mean accuracy: shared 93.33%, Mirror 93.84%, same-byte FiLM4 93.61%, FiLM8 94.01%, rank-1 95.07%, independent 97.20%. Mirror gains 0.51pp over shared, below the 1pp gate; at equal payload size (25,037B), FiLM4 is close and uses fewer correction MACs. Rank-1 is only 26,879B and more accurate. Independent bank is 84,787B.

## C — strongest counter-hypothesis

Any useful context specialization is captured by ordinary grouped affine or rank-1 output corrections; the four-angle rotation adds structure without enough quality gain.

## U — unresolved

No natural domain shift, larger language/vision model, adaptation-at-test measurement, or hardware latency claim.

## Evidence separation

- **Fact:** 168 rows replay with maximum metric difference 4.9e-9; payload hashes match and logits round-trip exactly.
- **Interpretation:** Mirror provides a small shared-model gain but no Pareto improvement over simple controls.
- **Hypothesis:** richer token-wise generated transforms may behave differently; MA-403 is a separate test.
