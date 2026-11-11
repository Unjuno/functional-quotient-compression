# MA-877 status

- Status: SCREENING (fresh evaluation pending)
- Branch: `research/ma-877-position-free-kv-transfer-20261111`
- Base commit: `66aa8caa`
- Protocol freeze: `79ff15fc`
- Selected rank: 32, frozen after development; this rank had the lowest dev NLL among tested ranks.
- Fresh/audit opened: no
- Results committed: pending
- Verification committed: pending

## Development screen

Position-free full maps were 0.617 nat above native target prefill versus 2.676 nat for direct RoPE-space ridge. Rank32 was 11.604 nat above full maps, while reducing marginal map payload from 2,996,100 B to 573,060 B (80.87%). Fresh gates are unchanged.

## Blockers

None for the frozen CPU-only, one-pair screen. CUDA is unavailable.
