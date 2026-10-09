# MA-585 status

- Status: FAIL (storage/quality frontier)
- Branch: `research/ma-585-mla-private-kv-residual-20261009`
- Base commit: `b115cb3a`
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

At zero heterogeneity, shared rank-2 cache used 2126B vs 2217B independent (4.1% fewer). At rho=.1, rank-2 attention-output nMSE averaged about .057; 50% private residual reduced this to about .001 at 3762B, exceeding independent storage, and full residual reached zero error at 4914B. Thus attention weighting requires private capacity that erases storage savings. Direct coefficients share the rank-2 basis; no Mirror-specific gain was found. Fresh sealed.
