# MA-556 status

- Status: FAIL (registered storage frontier)
- Branch: `research/ma-556-conditional-private-residual-20261009`
- Base commit: `e5916d18`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry and claim ledger, run verifier, and commit the development-only frontier result.

## Blockers

None.

## Decisions / rulings

Across two seeds, the oracle rank-2 basis has near-zero error at rho=0. At rho=.1, 10% private entries reaches mean nMSE about .00086 but costs 2548B versus 2539B for independent full operators. At rho=.3, 25% residual is borderline (one world exceeds 1e-3); 50% reaches mean .00015 at 3778B. At rho=.6, 50% reaches mean .000945 at 3778B. Direct coefficients match Mirror exactly. The registered 20% byte improvement over independent storage is not reached while passing the quality gate; fresh remains sealed.
