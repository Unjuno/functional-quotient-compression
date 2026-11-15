# MA-1031 — SAM2 interactive correction prompt views

Status: **FAIL**, scoped to three synthetic SAM2.1 Hiera-tiny videos. Mirror polar click history saves 7.5% versus direct float32 history, below the 25% gate, and loses 6.0 pp J&F. The native 8px grid is 185 B smaller and more accurate.

No click count is reduced: direct and Mirror use 40 clicks/video. The full session includes model, frame cache, prompts, per-object outputs and memory. Mirror saves only 43 B versus direct in that bundle.

- Frozen method/gates: `PROTOCOL.json`
- Development radii: `DEVELOPMENT.md`
- Fresh measurements and limitations: `RESULTS.md`
- Per-seed metrics: `RESULTS_CORE.csv`
- Input/history/prediction evidence and hashes: `runs/`, `VERIFICATION.json`
