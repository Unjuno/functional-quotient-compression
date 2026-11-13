# Amendment and execution notes

## Amendment 01 — before model/data download

Preregistered the real-language-model screen, model/dataset revisions, held-out ordered pairs, connector factorization/rank, bridge layers, update budget, seeds and metrics in `PROTOCOL.json`.

## Implementation corrections before any metric result

- GPT-2 tokenizer has no padding token; use EOS as pad and mask padded positions.
- Dropped the `low_cpu_mem_usage` load option because Accelerate is not installed; model weights and architecture remain pinned.
- Fixed hidden-state indexing and matching sequence length between mapped/native logits.
- First full evaluation attempted to hold full-vocabulary logits for all validation rows and exceeded the available process memory before emitting any metrics. Reworked evaluation to stream logits in batches; no result from the failed process was retained as evidence.
- Verified target-tail reconstruction independently: native target hidden states at layers 2/4/6/8 reproduce native logits exactly (max difference 0).

These corrections do not change the frozen split, model, connector, optimizer, training budget or quality gate.
