# MA-551 checked results

## Frozen screen and decision

The registered development seeds were run after protocol/source freeze and Amendment 1. Both seeds pass the quality and serialized-byte gates but fail the fused-kernel throughput gate; fresh seeds 55111–55113 remained sealed. No tuning followed measurement.

| Seed | Method | Payload bytes | Mean NRMSE | Max function NRMSE | Throughput / independent FiLM |
|---|---|---:|---:|---:|---:|
| 55101 | Shared FiLM + Mirror Givens | 2,994 | 1.17e-16 | 1.23e-16 | 0.693× |
| 55101 | Independent FiLM | 5,420 | 0.343 | 0.435 | 1.000× |
| 55101 | Native shared FiLM + Givens | 2,994 | 1.17e-16 | 1.23e-16 | 0.692× |
| 55101 | Rank-4 private residual | 20,288 | 0.224 | 0.350 | 0.305× |
| 55101 | Independent full affine | 36,150 | <1e-15 | <1e-15 | 0.182× |
| 55102 | Shared FiLM + Mirror Givens | 2,994 | 1.17e-16 | 1.24e-16 | 0.570× |
| 55102 | Independent FiLM | 5,420 | 0.329 | 0.436 | 1.000× |
| 55102 | Native shared FiLM + Givens | 2,994 | 1.17e-16 | 1.24e-16 | 0.667× |
| 55102 | Rank-4 private residual | 20,288 | 0.211 | 0.311 | 0.223× |
| 55102 | Independent full affine | 36,150 | <1e-15 | <1e-15 | 0.186× |

Tied FiLM uses 1,452 B, but max NRMSE is 0.455 in both worlds. Mirror is 0.552× independent-FiLM bytes and 0.0828× full-affine bytes. Its fused C kernel misses the 0.80× throughput gate in both worlds. The native Givens adapter has exactly equal serialized parameter arrays and exactly equal output errors.

Replay of seed 55101 reproduced all six NPZ payloads byte for byte. The exploratory seed-1 pilot is excluded from these results and recorded in Amendment 1.

## H / T / D / C / U

**H:** A shared FiLM object with a small functional view could recover 32 context functions.
**T:** Two development worlds; fused C kernels; six serialized controls; zero optimizer updates; actual payload bytes and CPU throughput.
**D:** FAIL, due the fixed runtime gate and exact ordinary Givens alias; fresh sealed.
**C:** The teacher and View share a conventional Givens code family, and computing these rotations is slower than FiLM despite kernel fusion.
**U:** Natural task benefit, learning from examples, GPU efficiency and other View geometries remain unknown.

**FACT:** Frozen gate outcomes and payload/metric replay are recorded per world in JSON and NPZ artifacts.
**INTERPRETATION:** This is aligned-family compression with an unfavorable tested runtime point and no Mirror-specific gain.
**HYPOTHESIS:** Lower-operation functional coordinates may avoid the rotation runtime cost.
