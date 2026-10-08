# MA-1097 status

Status: SCREENING — protocol frozen before reading adapter tensor values.

## H

A shared output-side B subspace gives lower held-out natural update error than a byte-matched shared input-side A subspace, while using at most 80% of exact independent LoRA serialized bytes at ≤10% update reconstruction error.

## T

Three public rank-1 BERT adapters (tweet_eval irony, emotion, hate), 12 layers × query/value, three whole-task leave-one-out folds, k=1/2. No optimizer updates or examples. Independent LoRA, B-shared, A-shared, and two-sided dense shared-core controls. Exact task/revision mapping and serialized bytes will be retained.

## D

Pending measurement.

## C

Any B/A difference is dataset-specific and fully explained by native shared-subspace projection; a native dense core or independent LoRA is more useful at the same payload.

## U

Exact training-time base revision and downstream task quality are not reported by this screen. No new-task adaptation or accelerator runtime claim.

## Fact / interpretation / hypothesis

- **Fact:** Adapter cards identify the same base model name, rank 1 and target query/value; they do not state the training-time base commit.
- **Interpretation:** A verified natural task-delta screen is feasible, but strict exact-base provenance is incomplete.
- **Hypothesis:** B-space directions generalize across these held-out task updates better than A-space directions.
