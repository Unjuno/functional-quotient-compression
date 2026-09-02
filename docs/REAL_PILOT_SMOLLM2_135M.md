# First Real Pilot Target: SmolLM2-135M

The first FQC real-model pilot target is `HuggingFaceTB/SmolLM2-135M`.

## Why this model

The model is small enough for repeated extraction/replay experiments while still
exercising architecture features that matter to FQC:

- Llama-style causal Transformer;
- grouped-query attention (9 query heads / 3 key-value heads);
- RoPE;
- SwiGLU/SILU MLPs;
- tied token embedding / language-model head;
- 30 Transformer layers;
- hidden size 576 and intermediate size 1536.

The public model repository is Apache-2.0. Configuration facts above were
checked against the public model metadata/config on 2026-09-02.

## Pilot staging

The pilot must not start by attempting a 64x codec over all layers.

1. Pin an exact model revision and compute hashes for the actual checkpoint
   files used. A repository commit ID is not a substitute for a file SHA-256.
2. Build the explicit Hugging Face Llama adapter plan from the pinned config.
3. Materialize the D57 manifest from live checkpoint tensors and verify the
   unique-storage baseline denominator.
4. Repeat extraction in a fresh process and require an identical public
   manifest hash.
5. Replay layer 0 attention/MLP primitives against the original module under a
   predeclared numeric replay contract.
6. Run D58 structural diagnostics and D61 functional-sensitivity probes on a
   small, fixed subset first.
7. Only after those checks pass, expand extraction and diagnostics across all
   layers and construct codec options for QCO.

## Evidence boundary

The model choice and adapter support are not compression results. The 64x target
remains untested on this model until an actual serialized paid payload and the
predeclared downstream-quality witness both exist.

## Current unresolved inputs

- exact Hub revision to pin;
- exact checkpoint file SHA-256 values;
- calibration/evaluation dataset and immutable sample IDs;
- replay tolerances for the selected execution dtype/backend;
- downstream quality metric/threshold.

These must be fixed before a real compression certificate attempt.
