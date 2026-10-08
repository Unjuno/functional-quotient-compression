# MA-498 status

- Status: **FAIL** (Mirror-specific attribution and frozen byte gate)
- Branch: `research/ma-498-code-distance-mirror-bank-20261008`
- Protocol frozen before development: yes (`freeze.json`)
- Development complete: yes (49801, 49802)
- Fresh/audit opened: **no** (49811–49813 remain sealed)

## Decision

At sigma .2, route error was 0.0004 in both learned-code seeds versus random-orthogonal 0.0010/0.0016 (robustness subgate passes). The learned payload was 9,384 B versus 8,842 B compact raw ID: ratio 1.0613, failing the frozen <=1.05 gate. Native metric learning exactly matches Mirror payload and every reported output metric. Classify FAIL; do not open fresh seeds.

## Fact / interpretation / hypothesis

- Fact: all methods replay eight functions without noise; the learned code covers all eight functions at every tested sigma.
- Fact: development payloads, full sigma curves and hashes are stored under `runs/dev_*`; independent replay max metric difference is zero.
- Interpretation: the code-distance benefit is ordinary metric learning and costs 542 B plus a larger decoder compute proxy compared with raw IDs.
- Hypothesis: geometry learned against a task-specific channel might improve a different native routing system, but this screen provides no evidence for it.
