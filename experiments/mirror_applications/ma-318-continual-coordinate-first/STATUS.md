# MA-318 status

- Status: FAIL — basis-growth mechanism works, Mirror storage/compute gate missed.
- Branch: `research/ma-318-continual-coordinate-first-20261008`
- Base commit: `594568f`
- Fresh 3/3: direct basis adds six directions, 2,160B, all-task max nMSE <=2.34e-7. Mirror adds the same six, max <=9.70e-6, but payload 2,654B (+22.9%) and fit proxy ~324x direct. FAIL.
- 144 checkpoints byte/hash checked and exactly replayed; four tests pass.
