# MA-546 status

- Status: SCREENING
- Branch: `research/ma-546-representation-symmetry-20261009`
- Base commit: `ac1c3e1814aada23eb99254b841faa31b4bb0ffb`
- Protocol frozen: yes
- Development complete: yes (amendment 1 corrected rerun)
- Fresh opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: pending

## Next action

Commit corrected development, replay and source freeze; then open fresh 54611–54613.

## Blockers

None.

## Amendment 1

First dev 54601 exposed the reversed dense compensation multiplication and a floating-point max-logit gate that overreacts to rare-token tail logits. Initial output is preserved under `results/pre_amendment_1/`. Dense compensation and distribution-level gate are corrected; rerun dev before fresh, which remains sealed.
