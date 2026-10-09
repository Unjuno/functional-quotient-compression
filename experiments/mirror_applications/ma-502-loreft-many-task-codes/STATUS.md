# MA-502 status

- Status: **FAIL for Mirror-specific attribution** (aligned quality/storage gate passes)
- Branch: `research/ma-502-loreft-many-task-codes-20261008`
- Protocol frozen before development: yes (`freeze.json`)
- Development complete: yes (50201, 50202; all nine task counts)
- Fresh/audit opened: **no** (50211–50213 remain sealed)

## Decision

The shared intervention crossed below 0.30x independent rank-four bytes at eight tasks, retained exact aligned outputs through 256 explicit task codes, and used 6,005 B versus 201,837 B independent. Native shared LoReFT coefficients exactly matched payload and outputs. Record FAIL for Mirror attribution; do not open fresh seeds.

## Fact / interpretation / hypothesis

- Fact: all 1–256 stored task codes were unique; measured functions had relative RMSE <=3.6e-6.
- Fact: code-zeroing changes outputs; independent rank-four and shared rank-four both use 196 operations/example.
- Interpretation: this is ordinary shared-basis coefficient storage, not new Mirror multiplicity.
- Hypothesis: natural or partially misaligned intervention banks may need an explicit private-state frontier.
