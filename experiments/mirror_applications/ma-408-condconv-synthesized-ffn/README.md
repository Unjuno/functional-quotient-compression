# MA-408 — CondConv-style synthesized FFN

## H — falsifiable hypothesis

For context-conditioned targets defined by a two-expert output mixture, synthesizing one effective FFN from a two-weight basis and then executing it once will stay within 1 percentage point of the output-mixture control while using at most 70% of its active MAC proxy and no more than 1.1× its serialized payload.

## T — protocol

Synthetic four-context classification. A fixed teacher blends two FFN expert logits with context-specific coefficients. Compare shared FFN, CondConv-style linear weight synthesis from two physical FFNs, output-mixture execution of two FFNs, context-specific rank-one residual, and independent context FFNs. Two development worlds select LR; three fresh worlds are locked. Measure actual serialized bytes, active compute including weight synthesis, wall clock, and per-context quality.

This tests CondConv as the native control for the subsequent Mirror basis-compression question. It does not attribute any gains to Mirror.
