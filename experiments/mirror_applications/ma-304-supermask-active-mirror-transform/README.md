# MA-304 — Supermask plus Mirror transform inside active edges

Status: protocol frozen before development. Dedicated branch: `research/ma-304-supermask-active-mirror-transform-20261008`.

## H — hypothesis

A single phase on two shared sparse edge bases will recover aligned functions without increasing the active edge count, and save at least 10% actual payload over the direct two-coefficient control. An unrelated-task bank tests the private-state boundary.

## Mirror insertion and controls

The base is a 32x32 linear map with exactly 256 active edges per task. Mirror interpolates two shared sparse bases using one task phase. Controls include dense shared weights, fixed sparse tying, a scalar gate, direct two-coefficient shared bases with identical private fallback, SupSup/Piggyback-style binary task masks over one dense backbone, and independent full matrices.

## Frozen protocol

See `PROTOCOL.json`. 128 aligned tasks and 16 unrelated tasks; 64 support, 32 validation and 64 test vectors per task; development seeds 30401/30402 and fresh seeds 30411/30412/30413. No optimizer updates. The actual deterministic ZIP/NPY payload includes all paid bases, indices, task codes, masks, private state and metadata.

## Development observations

Across both development seeds, Mirror used 21,288B versus 21,318B for direct coefficients (30B / 0.14% smaller), with 16 unrelated tasks privately stored in both. Mirror aligned max nMSE was 5.07e-5–5.26e-5; direct was about 1.5e-7. Mirror fit operation proxy was 452,984,832 vs 150,994,944 for direct (~3x). The frozen 10% fresh storage gate is therefore unlikely to pass, but fresh evaluation proceeds without tuning or changing gates.
