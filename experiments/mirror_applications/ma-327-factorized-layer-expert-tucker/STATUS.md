# MA-327 status

- Status: FAIL for Mirror-specific byte/quality value
- Branch: `research/ma-327-factorized-layer-expert-tucker-20261008`
- Protocol: `PROTOCOL.json` amended and pushed as `fd056ff` before fresh access
- Fresh seeds: 32711, 32712, 32713; all opened after protocol freeze
- Development complete: yes
- Fresh/audit opened: yes
- Results committed: yes (dedicated branch commit `4dde340ef71b561fd07c439b4044de5b92f6bd53`)
- Verification committed: yes; 15 payloads replay exactly, 5 tests pass
- Registry row updated: yes on integration branch

## Next action

Integrated on `research/mirror-application-current-evidence-20261008`; registry, claim ledger and status board record FAIL. No trained MoE/LM inference claim.

## Decisions / rulings

Mirror product and ordinary rank-2 coefficient product are the same function family. Their paired fresh payloads have identical SHA-256 and identical metrics in all three worlds. Both are 1,978B, 10.6% larger than the flat pair coefficient payload (1,788B). Do not characterize this as a Mirror-specific gain.
