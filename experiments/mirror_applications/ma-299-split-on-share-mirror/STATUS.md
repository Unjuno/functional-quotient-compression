# MA-299 status

- Status: SCREENING
- Branch: `research/ma-299-split-on-share-mirror-20261009`
- Base commit: `5408b693`
- Development complete: yes (worlds 29900–29901 × seeds 0–2)
- Fresh/audit opened: no
- Protocol locked: after this commit, before fresh

## Next action

Run frozen fresh worlds 29910–29912.

## Decisions

Changed sparse residual serialization to int16 indices and float32 values; residuals below 1e-6 are treated as reconstruction tolerance. Fixed split-count reporting for the never-split control. Changes occurred before fresh access.
