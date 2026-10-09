# MA-521 — Compile demonstrations into function-vector codes

Status: **FAIL** (frozen development screen; fresh seeds sealed). Branch: `research/ma-521-demonstration-to-code-20261009`.
Prior art: PA99 (causal Function Vectors) and PA82 (shared HyperFormer task-conditioned modules).

## H — Hypothesis

A demonstration-set encoder would predict a three-dimensional code for held-out relation functions, preserve explicit-FV query behavior, reduce stored payload and repeated demonstration tokens, and beat a same-rank linear compiler by at least 0.10 gold-logprob nats.

## T — Execution

We used pinned Pythia-70m, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, MA-516 task splits and FV extraction, 8 support pairs per task, task IDs 0–11 for compiler fitting and 12–15 held out, and development seeds 52101/52102. The encoder input was the L2-normalized mean of frozen Pythia token embeddings over the eight formatted support pairs. Rank-three linear and tanh encoders decoded through a shared 512x3 residual basis; both used exactly 2,000 Adam updates. Controls were no intervention, direct ICL with demonstrations repeated for each query, and the explicit FV bank. Fresh seeds 52111–52113 remain unopened.

The common model/config/tokenizer base was 168,144,624 B. Direct ICL token banks serialized to 5,962/5,986 B; explicit FVs to 34,450 B; each learned compiler to 17,248 B. The frozen half-bank limit was 17,225 B, so compilers exceeded it by 23 B. Direct ICL consumed 20,056/20,448 query input tokens, while compiled views consumed 2,072/2,016, reductions of 89.7%/90.1%. This token reduction does not assume a query amortization count. See RESULTS_CORE.csv for full bytes, support/query tokens, compute proxy, time, quality and reconstruction metrics.

## D — Decision

**FAIL** under the frozen gates. The tanh compiler lost 1.301/0.729 nats in held-out gold log probability versus explicit FVs; top-1 accuracy was .125/.250 versus .125/.219 explicit. It was 0.120/0.026 nats worse than the same-rank linear compiler, so there was no Mirror-specific benefit. Its 17,248 B payload exceeded the 17,225 B cap. Direct ICL had better gold log probability in both seeds (−9.619/−8.386) and a smaller serialized token bank. The only gate met was repeated-query context-token reduction. Fresh data stayed sealed.

## C — Strongest counter-hypothesis

Mean frozen token embeddings do not identify the task's causal function well enough for a three-dimensional code. The tanh compiler is a small conventional hypernetwork, and the same-rank linear control performed slightly better. Direct ICL preserves more function information while using less serialized state here.

## U — Boundaries

This is a four-task held-out constrained-recall compiler screen on one 70M model, not broad language understanding. Token savings depend on repeated use, while no query-count amortization is claimed. Other feature extractors, task families and model sizes are untested.

## Evidence classification

- **Facts:** frozen metrics and bytes are in RESULTS_CORE.csv. Three tests pass; all eight payload hashes, splits, teacher FVs and demo features replay exactly. Fresh seeds were not accessed.
- **Interpretation:** direct compilation saved query context tokens, but the code missed causal quality, strict storage and linear-control gates; this method did not beat direct ICL's stored token bank.
- **Hypothesis:** a more informative demo encoder or task-aware code may improve the tradeoff; this screen does not establish that possibility.
