# MA-307 status

- Status: SCREENING
- Branch: `research/ma-307-packnet-mirror-before-allocation-20261009`
- Base commit: `b8aff2b4`
- Development complete: yes (worlds 30700–30701 × seeds 0–2)
- Fresh/audit opened: no
- Protocol locked: after this commit, before fresh

## Next action

Run frozen fresh worlds 30710–30712.

## Development decisions

Fixed residual composition to add private corrections to shared predictions. Added robust generic shared-basis fitting because the initial PCA control used a mean-centered basis that was not comparable to the teacher's latent coordinates. The generic control estimates codes from majority inlier coordinates and preserves sparse outliers. Both changes preceded fresh evaluation.
