# MA-1039 status

- Status: SCREENING
- Branch: `research/ma-1039-plaid-mirror-pruning-20261115`
- Base commit: `720481fcc9fda56eb63483d3d42c7e000d8688f5`
- Last verified commit: `b03b1137f62be54aa24836091900b535e06b9194` (frozen protocol/source)
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

- Amendment before metric evaluation: raise the official CPU encoder microbatch from 16 to 128 and explicitly propagate `avoid_fork_if_possible=True` through both ColBERT config layers. The first run showed severe repeated CPU microbatch overhead; no data, split, model, index parameters, or quality settings changed. The interrupted run produced no index files or metrics.
