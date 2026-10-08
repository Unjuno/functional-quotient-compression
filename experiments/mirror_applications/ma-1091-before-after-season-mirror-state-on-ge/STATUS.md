# MA-1091 status

- Status: BLOCKED BEFORE PROTOCOL FREEZE; registry remains UNTESTED
- Branch: `research/ma-1091-seasonal-mirror-prithvi-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Last verified commit: pending
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: pending blocker record
- Registry row updated: no (correctly remains UNTESTED)

## Next action

Continue at the next uniformly sampled eligible MA candidate, Draw18 MA-784.

## Blockers

Available candidate dataset metadata does not establish seasonal labels and the documented Prithvi temporal path expects four frames while OSCD pairs contain two dates. Resolving either by substituting another estimand or inventing a frame construction would alter the registered experiment before its protocol is frozen.

## Decisions / rulings

- No data/model payloads were downloaded; no metrics were inspected.
- Draw17 was uniformly sampled from a frozen 550-row eligible pool at index 499 using seed `83b349a345f4a3289ebb44c7ba0ad7530337dcd16d4ecd1f4f2073699d599db3`.
- Draw18 uses a fresh uniform draw with its own frozen pool and replay record in `source/random_draw18.json`.
- This is NOT ESTABLISHED and is not a FAIL result.
