# MA-1035 status

- Status: SCREENING
- Branch: `research/ma-1035-qinco-mirror-view-20261115`
- Base commit: `e4ebfb8f7a1e45e3407fb1583c606ccd08cac81e`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Run frozen fresh worlds 103501–103503 without changing settings.

## Blockers

None. Faiss is unnecessary for this screen; exact PyTorch codeword assignment and exact brute-force L2 search are used.

## Decisions / rulings

A1 replaced unstable single-shot lookup timing with warm-up plus repeated medians before fresh worlds were opened. The initial timing artifact remains preserved. See `AMENDMENTS.md` and `FREEZE.json`.
