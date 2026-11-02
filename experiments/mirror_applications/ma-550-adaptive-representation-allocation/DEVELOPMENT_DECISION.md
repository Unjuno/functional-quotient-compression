# Development decision — MA-550

Both preregistered development worlds (55001, 55002) pass the adaptive support/held-out quality gates and beat both fixed payload sizes by more than 10%. The family allocation is 8 activation codes plus 8 sparse output-bias entries in each world.

| Seed | Adaptive bytes | Fixed all-activation bytes | Fixed all-weight bytes | Adaptive heldout relative RMSE | Fixed activation heldout RMSE | Fixed weight heldout RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 55001 | 19,682 | 34,496 | 3,220,842 | 0.0000329 | 0.4976 | 0.00224 |
| 55002 | 19,682 | 34,496 | 3,220,842 | 0.0000290 | 0.4974 | 0.00221 |

The all-activation baseline misses quality because it cannot represent the sparse one-token shifts; the all-weight dense baseline meets the 1% quality gate but costs 163.6× more bytes than adaptive. Adaptive's result is exactly a native sparse-bias plus ReFT/LoReFT-style shared output-basis code selector. This establishes a synthetic heterogeneous-allocation mechanism, not Mirror-specific value or natural task capacity. The fixed protocol authorizes opening fresh worlds 55011–55013.

Replay of seed 55001 reproduced SHA-256-identical NPZ files for all four serialized methods. No tuning followed development.
