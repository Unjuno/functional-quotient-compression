# MA-455 status

- Status: FAIL under registered quality/byte gates
- Branch: `research/ma-455-sequential-mirror-program-20261009`
- Base commit: `130db6b0`
- Initial protocol freeze: `08eb750e`
- A1 replacement fresh freeze: `eb08b0fd`
- Development complete: yes
- Valid fresh worlds: 45520–45522 × seeds 0–2
- Initial worlds 45510–45512: excluded exploratory; rank-one control had zero/zero initialization
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- H: Two sequential Mirror Views of one block can replace independent ordered blocks at materially lower bytes.
- T: 2D noncommuting rotation/shear products; 64 support and 512 query examples/task; tied, two-angle Mirror, rank-one step residual, independent blocks; 500 Adam updates. Actual payload sizes and order-swap checks at N=1/20/64.
- D (Fact): N20 average NRMSE: tied 0.000617, Mirror 0.0000154, low-rank residual 0.00000551, independent 0.0000000311. Bytes/task: 133.25, 177.65, 273.25, 193.85 respectively. Mirror is 8.4% smaller than independent, not the required 25%, and is far less accurate. Mean commutator 0.516; order-swapped NRMSE 0.319.
- D (Interpretation): Mirror can improve tied depth, but is dominated by independent quality and misses the storage gate. Tied blocks are smaller and strong on this task; Mirror does not provide the claimed replacement frontier.
- C: The teacher products are close to what a tied matrix can express; ordinary sharing explains most compression.
- U: Larger nonlinear models, longer programs, and natural-task results remain unknown.

## Next action

Commit experiment records, update authoritative registry and board, verify, push branch, then continue with next executable P0 MA-457.

## Blockers

None.
