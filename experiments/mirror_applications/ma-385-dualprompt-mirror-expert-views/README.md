# MA-385 — DualPrompt general prompt + Mirror expert views

Status: FAIL — development gates missed in both worlds  
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

## Report

**H:** A shared rank-two expert basis and one task angle can retain explicit DualPrompt task quality and reduce stored expert prompt bytes around a shared general prompt.

**T:** Two development worlds (38501, 38502), eight sequential tasks per world, 100 optimizer updates per task, 102,400 examples seen per method. Compared explicit general+expert prompts, hard-shared expert, direct rank-two coefficients, and Mirror-angle views. Fresh seeds 38511–38513 were not opened.

**FACT:** Actual serialized bytes were 1,352 B for explicit DualPrompt, 1,126 B for hard sharing, 1,424 B for direct rank-two coefficients, and 1,396 B for Mirror in both worlds. Mirror test accuracy was 0.8425/0.8425 versus 0.8730/0.8721 explicit, 0.8057/0.8015 hard-shared, and 0.8108/0.8074 direct-coefficient. Retrieval was 0.9993/1.0000. Mirror forgetting proxy was 0.093/0.095 versus 0.062/0.056 explicit and 0.113/0.128 hard-shared. The Mirror payload is 3.3% larger than explicit and 28 B smaller than the direct coefficient control. Serialized-state CPU throughput was below explicit in both seeds and close to direct coefficients.

**D: FAIL.** Mirror missed the 0.65x explicit byte gate, the 2-point explicit-quality margin, and the 5-point gain over hard sharing in both development worlds. Fresh remains sealed.

**C:** The shared general prompt and hard sharing already capture much of this aligned family. The rank-two expert view recovers some task accuracy, but the general prompt, router, and archive costs erase storage savings at eight tasks.

**U:** The data are synthetic and deliberately aligned. This screen does not test real images, natural class-incremental forgetting, or larger prompt pools, and fixed updates are not a near-convergence capacity result.

**INTERPRETATION:** This result does not establish useful prompt compression beyond ordinary shared-basis coding. Mirror slightly improves test accuracy over the direct coefficient screen with 28 fewer bytes, but remains below independent quality and exceeds the explicit bank size.

**HYPOTHESIS:** Larger task banks may amortize the basis and router cost; they require another registered test and cannot be inferred from this eight-task result.
