# Phase II roadmap — Sparse Shared-Rule / Mirror-MoE

Date: 2026-10-07
Status: active gated roadmap

## R0 — Repository/state consolidation

Goal: one current entry point, historical evidence preserved.

Exit:
- README points to current state;
- Phase II index distinguishes current vs historical;
- experiment registry exists;
- superseded architecture docs are explicitly marked;
- negative results include the major Phase II failures.

## R1 — Ordered functional composition

Question: can shared rule atoms represent genuinely ordered, non-commutative composition?

Construct tasks with:

[
A(B(x)) \neq B(A(x)).
]

Compare:
- independent sequential full experts;
- shared low-rank sequential rules;
- Mirror sequential rules.

Report:
- actual bytes;
- fixed-compute frontier;
- near-convergence frontier;
- held-out ordered compositions.

Exit:
- either repeated shared-rule advantage or explicit failure.

## R2 — Capacity vs learning-efficiency separation

For the best R1 architecture, sweep private-rule load and train substantially past the current 500–1500 update regime.

Primary outputs:
- retained rules vs bytes;
- retained rules vs training compute;
- convergence curves;
- active compute;
- wall-clock.

Exit:
- classify result as learning-efficiency-only, capacity-only, both, or neither.

## SRM003 active integration gate\n\nCombine R3 and R5 in one small causal Transformer: learn an overcomplete shared/private residual representation from LM loss only, discover private residuals from learned scores, physically prune, and compare to Dense / standard MoE / LoRA-MoE at actual bytes and active compute.\n\n## R3 — Routing discovery

Remove synthetic routing scaffolds in stages.

Compare:
- oracle;
- supervised;
- LM-loss-only with explicit rule tokens;
- LM-loss-only from ordinary contextual states.

Do not optimize load balance as a proxy objective unless it improves end quality.

Exit:
- stable routing advantage across fresh worlds, or identify routing as the limiting factor.

## R4 — Atom family competition

At matched bytes and compute compare:
- low-rank residual atoms;
- stretch;
- shear;
- combined stretch+shear;
- low-rank + Mirror hybrid.

Mirror remains only if it beats simpler shared-rule controls.

## R5 — Learn-many -> prune / rank allocation

Train overcomplete banks, then prune or reduce rank using validation-only criteria.

Compare to:
- same final bank from scratch;
- random pruning;
- usage-only pruning.

Measure quality recovery and actual bytes.

## R6 — Natural-language tiny-LM gate

Proceed only if R1–R4 produce a defensible synthetic frontier.

Use one small decoder Transformer first.

Compare:
- Dense;
- standard sparse MoE;
- shared low-rank rule MoE;
- Mirror rule MoE only if R4 justifies it.

Measure:
- validation NLL;
- targeted compositional probes;
- actual bytes;
- active FLOPs/MACs;
- throughput;
- training compute.

No rule-capacity claim is inferred from natural-language NLL alone.

## Stop / pivot criteria

- If near-convergence shared-rule capacity is not better and fixed-compute advantage disappears, reclassify the method as an optimization aid rather than capacity mechanism.
- If low-rank shared rules match Mirror, drop Mirror-specific complexity.
- If routing without explicit rule scaffolds collapses, keep representation and routing research separate.
- If natural-language quality requires independent-expert-like private storage, preserve the negative result and reconsider the factorization premise.
