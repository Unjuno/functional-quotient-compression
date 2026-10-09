# MA-583 status

- Status: FAIL (Mirror-specific gate)
- Branch: `research/ma-583-factorized-mla-mirror-20261009`
- Base commit: `0d4c52ff`
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

Across two seeds, factorized head×layer codes recover all six held-out roles exactly in 1437B vs 4389B flat. Direct factorized coefficients are exactly the same payload/output. Native shared MLA map is 600B but collapses to one distinct held-out role with nMSE .20159. This supports an aligned factorization mechanism, not Mirror-specific advantage or an MLA result. Fresh sealed.
