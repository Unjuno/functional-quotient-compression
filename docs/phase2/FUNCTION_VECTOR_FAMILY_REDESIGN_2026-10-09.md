# Function Vector application family redesign trigger

Date: 2026-10-09 UTC
Status: **pause new MA Function Vector execution until causal task utility is established**

## Trigger

The MA-516–522 screens repeatedly used prompt-minus-query residual differences as task Function Vectors. Multiple independent task formats and router/compression/persistence variants found that these deltas did not execute the requested behavior, even when direct ICL did. The failure is upstream of codebook, routing, quantization, or persistence quality. This meets the experiment contract stop rule for two consecutive candidates in one family failing from the same structural cause.

## Evidence (facts)

- MA-516: direct ICL 13.5–18.8% across worlds; query-only, explicit deltas and PCA codes were 0%.
- MA-517: direct ICL itself was 0%; task was invalid for FV composition; raw and layer-factorized codes were 0%.
- MA-518: direct ICL 31.25%; all prompt-delta interventions 0%. Factorized decoder had an implementation defect, so its factorization/byte result is not established.
- MA-519: context routing averaged 95.83% task-ID accuracy, but oracle and routed FV execution were 0%; router+FV was larger than explicit FV+ID.
- MA-520: direct ICL 25%; explicit/PCA/PQ interventions 0%. PQ reduced vector payload 50,921B→14,117B without task utility.
- MA-521: direct ICL 37.5%, code ID 25%, oracle/compiled FV 0%; compiled state exceeded explicit FV+ID bytes.
- MA-522: direct ICL 42.19%, while explicit/PCA/PQ persistent interventions were 1.56%; at eight turns PCA/PQ total state+context bytes also exceeded repeated ICL.

All results are scoped to pinned GPT-2 synthetic token tasks and this prompt-delta extraction interface. They do not falsify head-specific causal FVs from PA99 or natural language steering vectors from PA100.

## Interpretation

The common failure is not compression distortion alone: the uncompressed extracted delta does not preserve the desired task function. Router identity accuracy, vector NRMSE, and byte reduction therefore overstate functional utility if treated independently of task execution. Fixing compressors before fixing the teacher vector would repeat the same failure.

## Hypotheses for redesign (not yet tested)

1. The extraction interface is not causal: a last-token residual-state difference at one arbitrary layer is not the head-specific vector intervention in PA99.
2. GPT-2's next-token behavior on the synthetic templates may not support reliable direct execution, despite some direct-ICL scores above chance.
3. Task-label leakage and templated identity are easier than learning the transformation; router success is not evidence of a functional vector.

## Resume gates for MA-523 and later

Before testing shared/private residuals, a new development-only causal-vector screen must:

- select intervention sites/head outputs by a preregistered causal ablation/patching criterion, not arbitrary block choice;
- use task instances for which direct ICL reliably clears the task-specific chance baseline across development worlds;
- show an explicit, uncompressed intervention that improves the task-native held-out metric over query-only on development and an independent validation split;
- include layer/head ablations and a generic activation-addition control;
- freeze extraction, scale, prompt format, quality gates, and serialized-byte accounting before any fresh split is opened.

If hardware or suitable model weights are unavailable, record the family as BLOCKED/NOT ESTABLISHED and continue to the next non-FV registry family rather than relabeling synthetic vector reconstruction as task capacity. Existing MA-523 remains UNTESTED until a new amendment or redesigned protocol satisfies these gates.
