# MA-530 status

- Status: **FAIL**
- Branch: `research/ma-530-sae-feature-experts-20261009`
- Protocol froze before implementation and development.
- Development seeds 53001/53002 completed; fresh seeds 53011–53013 remained locked and unopened.
- Four tests pass; deterministic replay exact for ten paid payloads, core metrics, splits, router predictions and selected pool.

## H / T / D / C / U

- **H:** support-trained input routing over 16 SAE feature experts can approach same-router explicit FVs with smaller actual state.
- **T:** pinned Pythia-70m+SAE; 16 relation tasks; eight support and eight query examples each; standardized affine router; residual-pool64 OMP16; oracle/learned FV and global OMP16 controls.
- **D:** FAIL. Query-only task routing reaches .477/.516 accuracy, below .75. Shared-pool experts lose 1.013/1.162 nats and .117/.070 candidate accuracy to same-router FVs; total payload is 150 B larger than global sparse experts. Full SAE deployment adds 4,173,555 B over explicit routed-FV deployment.
- **C:** query-only input features may not disambiguate reverse/overlapping relation tasks; the poor route score is not a standalone SAE expert-capacity limit.
- **U:** explicit task context, other task families, natural MoE benchmarks, fresh data and optimized runtime remain unknown.

## Evidence classes

- **Facts:** replay maximum core metric difference 0; ten payload hashes exact; router predictions and pool exact; fresh unopened.
- **Interpretation:** learned routing is a major loss and shared feature coding adds further quality loss; the shared pool is also marginally larger than global sparse lists.
- **Hypothesis:** explicit task context or a different query-trained router could change the result; those require a new protocol.
