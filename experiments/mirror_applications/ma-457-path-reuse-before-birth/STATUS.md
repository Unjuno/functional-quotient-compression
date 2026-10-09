# MA-457 status

- Status: FAIL under quality, byte, and compute gates
- Branch: `research/ma-457-path-reuse-before-birth-20261009`
- Base commit: `0f3b102c`
- Frozen protocol: `b89b64fd`
- Development: threshold 0.1 selected from 0.01/0.03/0.1
- Fresh worlds: 45710–45712 × seeds 0–2
- Results/verification/registry: pending final commit

## H / T / D / C / U

- H: Mirror views reduce module births while retaining quality in continual PathNet-style reuse.
- T: 32-task synthetic stream (24 rotated shared functions, 8 off-orbit perturbations), 32 support + 32 route validation + 256 query points/task; path-only, Mirror view, and private modules. Actual payload and search costs measured.
- D (Fact): N32 births: Mirror 7.11, PathNet-only 24.22, private 32. NRMSE: 0.00769, 0.01522, 1.42e-7. Bytes/task: 71.82, 80.27, 69.28. Mirror search MAC 4.99M vs PathNet 0.10M per sequence; support-route wall 0.610s vs 0.0127s.
- D (Interpretation): Mirror cuts births and improves quality over path-only in this aligned screen, but misses the <=75% path payload gate and has much worse quality than private, with about 50× route-search MAC.
- C: Rotation-aligned tasks favor Mirror; cached path selection or natural task streams may behave differently.
- U: Natural continual tasks, learned high-dimensional blocks, cache-hit performance, and online update retention remain unknown.

## Next action

Commit verified evidence, update registry/ledger, push this research branch, then proceed to MA-461.

## Blockers

None.
