# MA-1120 status

- Status: FAIL
- Branch: `research/ma-1120-tntcomplex-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Last verified commit: `eeca3ab`
- Development complete: yes
- Fresh/audit opened: yes; final fixed-seed replay uses audit timestamps 10–11
- Results committed: yes (`eeca3ab`)
- Verification committed: yes (`eeca3ab`)
- Registry row updated: no

## Next action

Add the claim ledger entry after committing the checked result, then make a fresh random draw.

## Blockers

None.

## Decisions / rulings

- Two pre-result amendments fixed future timestamp extrapolation and made time features common to all methods.
- Audit aggregation initially included development timestamps; the fixed audit window was rerun with timestamps 10–11 only.
- Filtered MRR implementation was corrected and the same fixed fresh seeds were rerun.
- Final payload audit removed unused non-forward weights; clean payload results are authoritative.
- Independent timestamp-specific operators are not treated as an upper bound for unseen future times because those rows are untrained.
