# MA-431 status

- Status: SCREENING — development gate passed; fresh settings locked
- Branch: `research/ma-431-compositional-depth-view-20261009`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Protocol SHA-256: `c7b716913e042a65bea73a3293796687eddb5758e4dcd32b4acbcc3c98f1ab0c`
- Development: 72 rows; worlds 43100–43101 × seeds 0–2; 1500 updates per run
- Fresh/audit opened: no
- Frozen fresh settings: LR 0.001 for all methods; Mirror coordinate scale 1.0; worlds 43110–43112 × seeds 0–2

Development means: Mirror 6.07e-6 relative MSE / 2,921 B / 2,304 MAC proxy; independent 5.85e-6 / 3,945 B / 1,536; rank-1 gate 5.16e-4 / 3,425 B / 3,264; Universal low-rank 3.73e-6 / 3,993 B / 8,064. The half-angle setting failed the utility screen. See `DEV_SELECTION.json` and `results/development/DEV_GRID.csv`.

## Next action

Run only the frozen fresh settings; do not tune from fresh results.

## Blockers

None.
