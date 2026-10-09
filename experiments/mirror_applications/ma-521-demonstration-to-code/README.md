# MA-521 — Compile demonstrations into function-vector codes

Status: SCREENING; protocol fixed before implementation. Branch: `research/ma-521-demonstration-to-code-20261009`.
Prior art: PA99 (causal Function Vectors) and PA82 (shared HyperFormer task-conditioned modules).

## H — Hypothesis

A demonstration-set encoder can predict a three-dimensional code for held-out relation functions, preserving explicit-FV query behavior, reducing stored payload, and avoiding repeated demonstration tokens. The nonlinear code must also beat a same-rank linear compiler to earn a Mirror-specific result.

## Prior-art delta

PA99 establishes support-derived activation function vectors; PA82 establishes shared hypernetworks conditioned on small task/layer codes. Here the task address is predicted directly from the demonstration text using frozen token embeddings, then decoded to the layer-4 functional view. The direct-ICL context and a same-rank native linear compiler are mandatory controls.

## T — Frozen protocol

See `PROTOCOL.json` and `freeze.json`. Use pinned Pythia-70m, MA-516 relation tasks, 8 support pairs per task, tasks 0–11 to fit, and 12–15 held out. A frozen embedding mean is the demo-set feature. Compare no intervention, direct ICL, explicit FV, rank-three linear encoder-decoder, and rank-three tanh code encoder-decoder. Learned compilers use exactly 2,000 Adam updates. Development seeds 52101/52102; fresh 52111–52113 remain locked.

Charge the encoder, decoder, sixteen compiled codes, schema and direct-ICL token bank as actual uncompressed serialized bytes. Include the common model/config/tokenizer. Report support compilation tokens separately from repeated-query prompt tokens and avoid assuming an amortization count.

## D — Decision

Pending development.

## C — Strongest counter-hypothesis

A small generic hypernetwork or direct ICL may explain any benefit. Mean frozen token embeddings may not carry enough relation structure to predict the causal FV. The tiny held-out task set may make generalization noisy.

## U — Boundaries

This is a small constrained recall/compiler screen, not broad language understanding or open-ended generation. Held-out teacher FVs are used for evaluation only; code prediction sees only demonstrations.
