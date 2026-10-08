# MA-1097 status

Status: FAIL (weight-space screening; no downstream task claim)

## H

A shared output-side B subspace gives lower held-out natural update error than a byte-matched shared input-side A subspace, while using at most 80% of exact independent LoRA serialized bytes at ≤10% update reconstruction error.

## T

Three public rank-1 BERT adapters (tweet_eval irony, emotion, hate), 12 layers × query/value, three whole-task leave-one-out folds, k=1/2. No optimizer updates or examples. Independent LoRA, B-shared, A-shared, and two-sided dense shared-core controls. Exact task/revision mapping and serialized bytes will be retained.

## D

**FAIL.** Across all three whole-task holdouts, shared B/output-side k=1 had relative Frobenius errors 0.969–0.985; k=2 had 0.962–0.977. Shared A/input-side was consistently better (k=1: 0.910–0.938; k=2: 0.898–0.922), so the proposed B-side advantage reversed. One-sided views saved 29.9% of reserialized independent bytes at k=1 and 14.6% at k=2, but missed the ≤0.10 error gate by a wide margin. The dense two-sided core was smaller (38.4% / 69.2% of independent bytes) and also had near-total update error (0.991–0.999).

The exact independent bank used 482,216 serialized bytes; the original three adapter files plus their configs used 489,110 bytes. All methods included task classifier heads and metadata. No optimizer updates or examples were used. Output/input active MACs were 36,864 per token for the 24 rank-1 projections; dense-core k=1/k=2 used 36,888 / 73,824 per token. Factorized basis fits took about 0.003–0.029s per fold; reconstruction took 0.70–1.03s on CPU. This was not an end-to-end serving benchmark.

## C

Any B/A difference is dataset-specific and fully explained by native shared-subspace projection; a native dense core or independent LoRA is more useful at the same payload.

## U

Exact training-time base revision and downstream task quality are not reported by this screen. No new-task adaptation or accelerator runtime claim.

## Fact / interpretation / hypothesis

- **Fact:** Adapter cards identify the same base model name, rank 1 and target query/value; they do not state the training-time base commit.
- **Fact:** 19 serialized payloads replayed with equal bytes, keys and tensor values; maximum rank-1 gauge/projector drift was 4.86e-17. Four focused tests pass.
- **Interpretation:** These three adapter deltas do not support B-side shared-view compression at the frozen reconstruction threshold. A-side projection is better in this task bank, but remains far outside useful update fidelity.
- **Hypothesis:** Task specialization in the q/v LoRA factors is largely independent across this tweet_eval bank; exact-base confirmation and downstream validation would be needed to extend that conclusion.
