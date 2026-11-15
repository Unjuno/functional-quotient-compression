# MA-1030 status

- Status: SCREENING
- Branch: `research/ma-1030-sam2-rare-object-private-fallback-20261115`
- Base commit: `fc7c2640fef0da03f7064d3d41e4b20f3afc2545`
- Protocol: frozen in commit `ec729acfe`; development seed 103000 complete; selected threshold 0.15 committed before fresh data generation
- Audit seeds 103001–103003: not generated
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

The dev-selected threshold and full development record are committed. Generate only the three fresh audit worlds, then run the frozen controls.

## Development finding

The frozen residual-energy selector picked object ID 42 on all six frames in development; rare shifted object ID 99 was never selected. Keep this as a pre-audit warning and evaluate unchanged.

## Blockers

None known. Official SAM2.1 Hiera-tiny source/checkpoint are present in the external research cache.
