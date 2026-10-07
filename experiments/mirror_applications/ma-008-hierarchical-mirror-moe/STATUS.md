# MA-008 status

- Status: PROMISING for aligned expert sharing
- Branch: `research/ma-008-hierarchical-mirror-moe-20261007`
- Protocol/source freeze: `5ad3afe`
- Development selected LR 0.003 only on world 80000
- Fresh worlds 80001–80003 complete for both teacher modes
- Replay: all 42 fresh deterministic rows match exactly
- Verification: 3 tests passed; serialization round-trip checked across 84 runs
- Registry: PROMISING

## H / T / D / C / U

- **H:** A hierarchical group→expert router plus one nonlinear FFN and per-role views can preserve related expert behavior at lower bytes than untied hierarchy.
- **T:** Seven methods, two teacher modes, 1,200 updates × 64 examples, development world 80000 and fresh worlds 80001–80003.
- **D:** PROMISING: useful-sharing passed 3/3; Mirror/full-hier MSE 0.931–0.949 and payload 0.361x. Hard tying is 317B smaller; Mirror CPU throughput is 0.317x tied. Full hierarchy did not consistently beat flat.
- **C:** Givens-aligned synthetic teacher and clean balanced hierarchy favor this mechanism; generic shared bases could explain the result.
- **U:** Language transfer, capacity, larger hierarchies and optimized inference remain untested.
