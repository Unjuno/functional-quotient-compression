# MA-004 status

- Status: SCREENING
- Branch: `research/ma-004-soft-mirror-expert-mixture-20261007`
- Base commit: `7ff5430` (verified MA-002 parent)
- Protocol frozen before development; no data opened
- Registry: UNTESTED

## Scope ruling

All four nonlinear expert outputs are evaluated and mixed by learned full-softmax router weights. This differs from sparse top-2 MA-002 and deterministic signed linear MA-005.

## Next action

Adapt the frozen top-2 harness for dense soft routing, add correctness tests, then freeze implementation before development. Fresh worlds remain sealed unless development passes.
