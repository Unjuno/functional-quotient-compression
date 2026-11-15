# MA-1039 status

- Status: SCREENING
- Branch: `research/ma-1039-plaid-mirror-pruning-20261115`
- Base commit: `720481fcc9fda56eb63483d3d42c7e000d8688f5`
- Last verified commit: pending protocol/source freeze
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Commit the frozen source, tests and protocol before downloading or inspecting the TREC-COVID archive.

## Blockers

None known. Official CPU PLAID imports and the required FAISS/C++ toolchain are available; an index build/search smoke run remains to be performed.

## Decisions / rulings

- Fixed all arms at `ndocs=4096`; the official searcher otherwise supplies different `ndocs` defaults by requested k, which would confound the conditions.
- Raw role length is counted with the pinned ColBERT query tokenizer.
- One-pass timing is reported as NOT ESTABLISHED, not as a stable latency comparison.
