# MA-626 — Shared canonical SDF decoder with per-shape Mirror code

Status: SCREENING; prior art PA131. Dedicated branch: `research/ma-626-deepsdf-mirror-autodecoder-20261103`.

## H

For a family of translated and scaled 2D circles, a shared canonical signed-distance decoder plus a 3-value Mirror coordinate should reproduce held-out functions with fewer actual inference bytes than independent per-shape fields. A Mirror-specific advantage requires beating direct native shape conditioning at matched quality, bytes, and compute.

## T

For each development seed, generate 16 circles with centers and radii from fixed ranges; hold out four by a frozen split. Evaluate 64 support and 256 disjoint audit points per shape. Mirror method decodes `m=(cx,cy,r)` through the canonical unit-circle SDF. Controls: direct native analytic shape conditioning with the same three values; independent 64×64 FP32 SDF grids; shared canonical decoder with stored per-shape codes. Measure actual serialized payload bytes and evaluation operations. Fresh seeds remain sealed after an exact native alias.

## D

Pending.

## C / U

Counter-hypothesis: Mirror is only a coordinate reparameterization of the ordinary analytic shape conditioner. This restricted geometric mechanism does not test learned DeepSDF decoder training or general shape families.
