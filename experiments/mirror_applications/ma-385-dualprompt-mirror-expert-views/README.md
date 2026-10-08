# MA-385 — DualPrompt general prompt + Mirror expert views

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME / RETENTION  
Base commit: `0c3580b` (MA-383 result + MA-374 replay audit)  
Prior art: PA57 (DualPrompt)

## Hypothesis

H: When task-specific prompt residuals lie near a shared two-vector orbit around one task-invariant general prompt, one shared expert basis plus one angle per task can retain DualPrompt task quality and reduce actual serialized bytes versus storing every expert prompt; any Mirror-specific claim must also beat a direct two-coefficient basis control.

## Mirror insertion

> **Mirror insertion:** this experiment adds a task-specific angular coordinate `m_j` to the expert-prompt residual component of DualPrompt so that multiple task expert prompts can be reconstructed from one physical two-vector basis around the shared general prompt.

- Native method: one shared general prompt and one explicit expert prompt per task.
- Mirror: one shared general prompt plus `cos(m_j)u + sin(m_j)v` for each retrieved task.
- Persistent m: one FP16 angle per task.
- Direct controls: the native explicit expert bank, hard sharing of one expert residual, and rank-two expert basis with direct task coefficients.
- Retrieval is top-1 cosine over the same train-key centroids for all methods; no task ID is supplied at test time.

## Physical-to-logical claim

The general prompt is the task-invariant component. The expert residual carries task-specific behavior. The synthetic teacher family includes a common decision component plus aligned two-dimensional task residuals, so favorable results are only aligned-family feasibility. The screen records final task accuracy and retention across the sequential task schedule; it is not a natural DualPrompt or image benchmark.

## Comparisons

1. DualPrompt-style explicit general prompt + one independent expert prompt per task.
2. General prompt + one hard-shared expert residual.
3. General prompt + shared rank-two expert basis + direct two-coefficient task codes.
4. General prompt + shared rank-two expert basis + one angular Mirror code per task.

All methods use the same train order, task keys, updates, examples, and frozen feature representation. The router and all prompt state are paid in the inference payload.

## Gates

### PASS (development screen)

Both development worlds: explicit DualPrompt test accuracy at least 0.85; Mirror within 2 percentage points of explicit accuracy; Mirror actual serialized payload no more than 0.65x explicit and no more than 1.10x the direct rank-two basis control; task retrieval at least 0.98; Mirror improves mean final accuracy over hard-shared experts by at least 5 points; retention/forgetting and compute are reported. Fresh worlds run only if every condition passes in both worlds.

### FAIL

Either development world misses any registered quality/storage/retrieval threshold or the direct coefficient control matches Mirror at comparable or lower bytes. Fresh remains sealed.

### NOT ESTABLISHED

Serialization, deterministic replay, or runtime drift prevents interpreting a frozen result. No fresh result may repair a development failure.

## Tuning boundary

- Development worlds: seeds 38501 and 38502.
- Fresh worlds: seeds 38511, 38512, and 38513; sealed unless both development worlds pass every gate.
- Only implementation details needed to fit the frozen model/task definition may be corrected on development data. No architecture or gate changes after looking at results.

## Storage and compute

Actual deterministic ZIP/NPY FP16 payload bytes are authoritative. Count the general prompt, all explicit expert prompts or shared basis, task codes, router keys, and metadata. Report examples/updates, training time, estimated router/prompt/classifier MACs, serialized-state CPU throughput, per-task accuracy, and forgetting.

## Scope

This is a synthetic continual-task prompt mechanism screen. It does not establish natural image accuracy, class-incremental robustness, or production DualPrompt behavior.
