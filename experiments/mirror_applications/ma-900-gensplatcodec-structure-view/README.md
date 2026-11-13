# MA-900 — GenSplatCodec structural × appearance Mirror code

Status: **UNTESTED — blocked before model evaluation**
Evidence lane: NEURAL_GRAPHICS / GENERATIVE_COMPRESSION / MULTI_VIEW_QUALITY / STORAGE / RUNTIME
Base commit: `b6bc773651fff85036e6612a30202817b9998c32`

## H — falsifiable hypothesis

A compact structural × appearance View code can recover multiple useful logical scene variants from shared GenSplatCodec state while improving actual stream rate, multi-view quality, and latency over native and simple latent/gain controls. The hypothesis fails if these controls match the gain, quality or consistency gates fail, or total stream/decode cost eliminates savings.

## Prior-art delta

PA265 already combines geometry and appearance streams with a geometry-guided one-step generative decoder. The Mirror claim is only incremental value beyond that native design and ordinary code/photometric controls.

## Execution limits

See `EXECUTION_BLOCKER.md` and `source/ENVIRONMENT.md`. Bounded source search found no official implementation; local runtime lacks CUDA and the relevant diffusion/rasterizer stack and scene assets. No model or codec was run. This is a feasibility review, not a negative result.

## Report

H/T/D/C/U are recorded in the blocker report. Fact: no model execution. Interpretation: registered frontier is not established. Hypothesis: Mirror value over native structure/appearance coding remains untested.
