# MA-929 execution blocker

## Fact

- PA240 is *StitchLLM: Serving LLMs, One Block at a Time*, ACL 2025. It uses trainable stitching layers and routing under a serving cost-quality tradeoff.
- ACL Anthology page was inspected. Exact-title GitHub repository search returned zero repositories; Hugging Face model search returned no results.
- No matched source/target LLM checkpoints, native stitching connector bank or serving evaluation assets are available locally.
- Registry explicitly requires strong native StitchLLM throughput control.

## Decision

Keep UNTESTED. Substituting nanoGPT or another unrelated stitching implementation would remove the required native control and change the candidate. No synthetic connector result would establish NLL/throughput.

## Resume condition

Obtain official code and model/connector assets, reproduce native StitchLLM, then freeze ordered source/target block splits and measure NLL, exact connector bytes and serving throughput against shared-basis and Mirror controls.
