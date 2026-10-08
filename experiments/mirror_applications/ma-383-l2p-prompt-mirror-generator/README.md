# MA-383 — L2P prompt pool with Mirror prompt codes

Status: FAIL — development gate missed in both worlds  
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

One physical rank-two prompt basis is intended to express four logical task prompts. In this minimal linearized proxy, a retrieved prompt supplies a task-specific linear readout over a fixed identity feature representation; it approximates the local functional effect of one soft prompt, but it is not a frozen-backbone L2P benchmark. A fixed synthetic task family generates teacher boundaries on a two-dimensional rotation orbit; this makes a favorable result a scoped aligned-family result, not natural continual-learning evidence.

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

## A1 serialization audit

The first development pass was preserved unchanged under `results/development/`. Its evaluator used FP32 router centroids while the serialized inference payload stored them as FP16. A1 repeats the same development seeds and optimization schedule and evaluates after reloading the quantized router and prompt state from each actual payload. A1 does not change gates or expose fresh seeds.

## Report

**H:** A shared rank-two prompt basis and one angular code per retrieved task will retain explicit-prompt quality, save actual bytes, and beat hard sharing and a direct coefficient control.

**T:** Two development worlds (38301, 38302), four sequential synthetic tasks per world, 600 updates per method, and 76,800 examples seen per method. Compared explicit prompts, hard sharing, direct rank-two coefficients, a key-conditioned generator, and the one-angle Mirror basis. Fresh seeds 38311–38313 were not opened.

**FACT:** After loading FP16 prompt and key tensors from actual ZIP/NPY payloads, Mirror uses 1,034 B versus 838 B for independent prompts, 788 B for hard sharing, 1,054 B for direct coefficients, and 2,092 B for the hypernetwork. Mirror test accuracy was 0.852/0.833 versus 0.937/0.929 independent, 0.831/0.823 hard-shared, and 0.827/0.803 direct-coefficient. Retrieval was 0.993/0.999 for all methods. Mirror missed the 0.60x independent byte gate, missed the 2-point quality margin, and did not improve hard sharing by 5 points. CPU throughput varied by seed; no stable runtime gain was measured.

**D: FAIL.** Multiple frozen development gates were missed in both worlds. Fresh remains sealed.

**C:** The task family is small and aligned, but the angular constraint and sequential shared-basis optimization still lose task-specific boundary quality; router and archive overhead exceed the savings from four small prompts.

**U:** This proxy does not establish behavior on image prompts or natural class-incremental L2P. A one-token linearized prompt cannot answer whether multi-token prompts over a pretrained backbone behave differently.

**INTERPRETATION:** These results do not support Mirror-specific compression for this setting. The direct two-coefficient basis achieves similar or worse quality at nearly identical payload size; the one-angle code saves only 20 B versus that direct control and still exceeds the independent prompt bank.

**HYPOTHESIS:** Larger prompt banks may amortize the shared basis and router costs, but require a new registered experiment and fresh evidence.
