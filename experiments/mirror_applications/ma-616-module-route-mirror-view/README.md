# MA-616 — joint route and module View factorization

Status: SCREENING. Prior art PA127: Routing Networks and PathNet.

## Hypothesis
**H:** on tasks reusing a small module bank through Givens-conjugated Views, route IDs plus a small phase code reproduce aligned functions with less serialized state than direct coefficients; unrelated functions expose the private-state boundary.

## Frozen protocol
Two-stage 2×2 ReLU functions; four physical modules per stage; 32 tasks per world (24 aligned View tasks, 8 unrelated tasks). Compare route-only PathNet, route × one-angle Mirror View, route × direct sine/cosine coefficient control, and independent per-task matrices. Development seeds 61601/61602; fresh seeds 61611–61613. No optimizer updates. All banks, routes, codes, private fallback matrices and metadata are paid. Router uses oracle task IDs; no learned routing claim. CPU timing includes code decode/materialization and 256 queries.

Facts, interpretation and hypothesis are reported separately in STATUS.md. Synthetic mechanism screen only.
