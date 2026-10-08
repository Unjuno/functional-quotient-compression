# MA-383 — L2P prompt pool with Mirror prompt codes

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `0739bf3` (MA-381 result branch)  
Prior art: PA56 (L2P)

## Hypothesis

H: For task families whose useful prompts lie near a shared two-vector orbit, storing a shared prompt basis plus one angle per retrieved task preserves prompt-conditioned classification quality with materially fewer actual serialized bytes than an explicit L2P-style prompt bank, while beating hard sharing; it must also beat a direct two-coefficient basis control to support any Mirror-specific claim.

## Mirror insertion

> **Mirror insertion:** this experiment adds one angular coordinate `m_j` to a shared two-vector prompt basis so that each retrieved task can use a different logical prompt without storing a complete task prompt vector.

- Native method: retrieve one explicit trainable prompt per task from a prompt pool.
- Insertion: replace each stored prompt vector with `cos(m_j) u + sin(m_j) v`.
- `m_j` is persistent and stored with the prompt payload.
- Cheapest controls: one hard-shared prompt, a learned rank-two prompt basis with two direct coefficients per task, and a small key-conditioned hypernetwork.
- Retrieval uses the same learned key prototypes and cosine top-1 rule in every condition; no task ID is supplied at test time.

## Physical-to-logical claim

One physical rank-two prompt basis is intended to express four logical task prompts. Each prompt is added to a frozen shared feature representation and read by a frozen shared binary classifier. A fixed synthetic task family generates teacher boundaries on a two-dimensional rotation orbit; this makes a favorable result a scoped aligned-family result, not natural continual-learning evidence.

## Comparisons

1. Independent explicit task prompts (upper control).
2. One hard-shared prompt.
3. Shared rank-two prompt basis with two unconstrained coefficients per task.
4. Key-conditioned two-layer prompt generator.
5. Shared rank-two prompt basis with one angular Mirror code per task.

Every method receives the same examples, router, updates, key pool, and frozen feature/readout maps. Keys, generator, basis, codes, metadata, and archive overhead are included in serialized bytes.

## Gates

### PASS (development screen)

Both development worlds: explicit prompt accuracy at least 0.80; Mirror accuracy within 2 percentage points of explicit prompts; Mirror actual payload no more than 0.60x explicit prompts and no more than 1.10x the direct rank-two coefficient control; retrieval accuracy at least 0.98; Mirror beats hard sharing by at least 5 percentage points; training and inference MAC proxies are reported. Fresh worlds are only run if this entire gate passes.

### FAIL

Either development world misses any registered quality/storage/retrieval threshold, or the direct coefficient control matches Mirror quality at comparable or lower bytes. The latter specifically falsifies Mirror-specific advantage while leaving aligned prompt compression as an ordinary low-rank result.

### NOT ESTABLISHED

Replay, serialization, or environment drift prevents deciding the frozen screen; no fresh results may be used to repair a development failure.

## Tuning boundary

- Development worlds: seeds 38301 and 38302; only these may set optimizer details within the frozen architecture.
- Fresh worlds: seeds 38311, 38312, and 38313; sealed unless both development worlds pass all gates.
- Synthetic one-token prompt proxy only; no claim about L2P continual image benchmarks or real class-incremental learning.

## Storage and compute

Actual deterministic ZIP/NPY FP16 inference payload bytes are authoritative. Charge prompt weights, all per-task codes, router keys, hypernetwork state, reconstruction metadata, and archive headers. Report training examples/updates, train wall time, inference MAC proxy, retrieval/prompt-generation operations, and examples per second.

## Decision language

Separate observed facts, interpretation, and hypotheses. A code count is not a count of independent capabilities. Any positive result is only an aligned synthetic mechanism result until independently replicated on a natural continual-learning benchmark.
