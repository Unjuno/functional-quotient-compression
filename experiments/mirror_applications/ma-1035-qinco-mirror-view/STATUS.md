# MA-1035 status

- Status: FAIL
- Branch: `research/ma-1035-qinco-mirror-view-20261115`
- Base commit: `e4ebfb8f7a1e45e3407fb1583c606ccd08cac81e`
- Development complete: yes
- Fresh/audit opened: yes, worlds 103501–103503
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Commit and register this scoped FAIL, push the branch, then continue to the next eligible P0 (MA-1036; MA-1034 is P1).

## Blockers

None. Faiss was unnecessary for this scale; exact PyTorch codeword assignment and exact brute-force L2 search were run.

## Decisions / rulings

A1 replaced unstable single-shot time measurements with warmed repeated medians before fresh-world access. The first timing artifact is preserved. See `AMENDMENTS.md`.
