# MA-533 — Native skip-transcoder MLP approximation fidelity

## H — Hypothesis

The released 128x top-k skip-transcoder for SmolLM2-135M layer 8 approximates the native MLP output on held-out natural text activations with relative MSE at most 0.10 and mean token cosine at least 0.95. This is a prerequisite screen for MA-534, not evidence of Mirror benefit or logical multiplicity.

## T — Protocol

The experiment uses the pinned public checkpoint `EleutherAI/skip-transcoder-SmolLM2-135M-128x` at revision `651f51421f2e1aa8fbd907e02ef421d3da55ff6d`, model `HuggingFaceTB/SmolLM2-135M`, layer 8, and deterministic 128-token windows from the Wikitext-2 training split. Sixteen windows fit a rank-128 linear cross-covariance SVD control; sixteen disjoint windows evaluate native MLP vs transcoder. Development is CPU-only, one thread, no gradient updates. Fresh seeds 53311–53312 stay sealed unless development passes.

Controls are native MLP target, transcoder skip-only, native top-128 transcoder and rank-128 linear low-rank control. Exact safetensors and config byte lengths are recorded; base-model files are charged for standalone deployment. Active top-k, token count, wall time, and MAC proxy are reported.

## Current result

SCREENING; protocol/source are being frozen before data evaluation.

## Scope

A successful screen validates one released transcoder checkpoint/layer only. MA-534 must still show functional-role quality over plain sparse feature gating and strong low-rank/gating controls.
