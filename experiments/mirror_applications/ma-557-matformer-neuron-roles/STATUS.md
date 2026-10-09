# MA-557 status

- Status: FAIL (Mirror-specific gate)
- Branch: `research/ma-557-matformer-neuron-roles-20261009`
- Base commit: `7ede5c3e`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry and claim ledger, replay verifier, and commit the development-only FAIL.

## Blockers

None.

## Decisions / rulings

Two seeds: Mirror role gains and direct ordinary gains both use 1016B and exactly recover the three width targets. Independent width FFNs use 1249B. Plain nested prefix uses 739B but mean nMSE is .06485. The native-tail synthetic serialization used 1409B and is an upper cost control, not evidence that production MatFormer tails cost this much. Fresh remains sealed because the direct gain control is exact.
