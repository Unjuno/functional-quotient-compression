# MA-452 status

**FAIL — factorized addressing generalizes, but Routing Networks matches quality with smaller addresses and the same role search.**

## H / T

Tested a discrete path factor and support-inferred role factor on 8 held-out combinations from a 4-path × 6-role grid. The route ID is provided and charged; the role is inferred from 32 support examples. Three fresh worlds × three seeds, 64 tasks per seed.

## D — Fact

At N=20, mean held-out NRMSE was Mirror 0.00e+00, Routing 0.00e+00, PathNet-only 0.197, and flat full matrices 1.92e-07. Actual bytes/task: Mirror 145.45B, Routing 139.05B, flat 146.05B. At N=64, Mirror 48.45B vs Routing 44.45B and flat 89.64B. Both Mirror and Routing use 8 physical modules and the same six-candidate support search.

## Interpretation

Factorized path-role codes recover unseen combinations and beat the flat independent table slightly at N=20. The role coordinate is not Mirror-specific: a one-byte Routing Networks role index gives the same function with fewer serialized bytes and comparable compute.

## C — Strongest counter-hypothesis

Routing Networks already composes reusable modules and role choices; Mirror encodes the same discrete role as a 32-bit angle and pays extra address bytes.

## U — Unknown

Continuous roles, learned route discovery, nonlinear networks, and natural tasks remain untested. The route identity is supplied here.
