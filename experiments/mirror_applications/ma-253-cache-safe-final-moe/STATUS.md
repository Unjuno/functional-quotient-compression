# MA-253 status

- Status: FAIL (Mirror expert replacement); cache-placement mechanics PASS
- Branch: `research/ma-253-cache-safe-final-moe-20261007`
- Base commit: `01b515cf8d0603481cb00ff1d5be45541411eebb`
- Last verified commit: `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`
- Development complete: yes; common LR `0.003`
- Fresh/audit opened: yes; worlds 25301, 25302, 25303
- Results committed: yes (`1891cbc36d3a99b4dd63517b469f8b246dbf0be0`)
- Verification committed: yes (`1891cbc36d3a99b4dd63517b469f8b246dbf0be0`)
- Registry row updated: yes (FAIL; independent rank-2 domain updates)

## Decision

The Givens-view model saves 28.1% of payload bytes relative to the DMoE rank-2 LoRA control but fails the preregistered quality gate in 3/3 fresh worlds. Rank-1 private residuals recover a large fraction of the error but do not match rank-2 LoRA. The independent rank-2 control reaches the synthetic teacher. Final-only placement has exact cache invariance; a view placed in an early layer changes downstream K/V.

The implementation also has a training-time slowdown: 9.6–11.9s for Mirror versus 1.3s for rank-2 LoRA at equal update count. CPU inference throughput is variable and is recorded in verification.

## Next action

Push the dedicated MA-253 branch, then continue to MA-244.

## Blockers

None.

## Decisions / rulings

- FAIL is scoped to this Givens view on independently generated rank-2 domain updates, not all Mirror coordinates.
- Treat supplied domain address as an explicit task input and exclude router cost from the claim.
- Preserve the distinct cache placement result and compute slowdown alongside the quality/storage failure.
