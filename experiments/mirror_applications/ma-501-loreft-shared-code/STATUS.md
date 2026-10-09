# MA-501 status

- Status: **FAIL** (rho=.25 quality gate; no Mirror-specific attribution)
- Branch: `research/ma-501-loreft-shared-mirror-code-20261008`
- Protocol frozen before development: yes (`freeze.json`)
- Development complete: yes (50101, 50102)
- Fresh/audit opened: **no** (50111–50113 remain sealed)

## Decision

At rho=0, shared rank-four codes are lossless at 2,138 B versus 13,668 B independent rank-four; native shared coefficients exactly match the candidate. At rho=.25, shared heldout relative RMSE is 0.226/0.173, failing <=0.10; independent rank-four is 0.178/0.161, while the full-matrix upper is exact. Do not open fresh seeds.

## Fact / interpretation / hypothesis

- Fact: the heldout task codes causally change outputs; zeroing/permuting them changes predictions by max 0.344–0.469 in the two worlds.
- Fact: native shared-subspace coefficients have exact serialized payload and output aliases across all four rho values.
- Interpretation: the aligned saving is ordinary shared low-rank intervention; private variation degrades quality and eventually calls for extra rank or private state.
- Hypothesis: natural LoReFT task deltas may admit a useful shared/private frontier, but that was not tested here.
