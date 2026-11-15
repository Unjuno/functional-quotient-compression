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

- Amendments before metric evaluation:
- Explicitly propagate `avoid_fork_if_possible=True` through both ColBERT config layers. This avoids a multiprocessing launch and leaves official CPU scoring semantics unchanged.
- Use official `index_bsize=64` (the first run used 16; the 128 trial was slower). A 64-document microbenchmark gave bit-identical document embeddings at batch sizes 8, 16, 32 and 64; observed times were 10.016, 10.504, 12.054 and 10.164 s respectively. Batch 64 is the pinned official default.
- The upstream automatic codebook sample would encode all 18,049 passages before encoding all of them again into the index. Bound codebook training to a seeded sample of 4,096 passages, all candidates remain in the final index. This is a compute-only screening deviation, documented in PROTOCOL; native and Mirror arms share the exact resulting index. No quality metrics have been viewed.
- Both interrupted index runs produced no saved index chunks or metric outputs.

- Recovery amendment: the first completed development+audit pass reached final summary generation but failed because the driver hashed `DATA/trec-covid.zip` instead of `DATA.parent/trec-covid.zip`; no audit results were printed or persisted. Preserve the development-only angle freeze and replay only the audit after fixing archive hashing, complete checkpoint file accounting (including artifact metadata), and index-build timing recovery. This is not a tuning amendment; no audit metrics were inspected.

- Second deterministic replay reached audit completion but hit a final report-only `angle_scores` scope error when development freeze reuse skipped the sweep. No metric output was shown or stored. The summary now reads the frozen development MRR array directly; audit arms remain unchanged.

- Recovery/report amendment: persist per-query audit top-1,000 rankings and pre-pruning candidate IDs so quality and candidate-union recall can be independently replayed; no training, split, angle, index, or scoring parameter changes.
