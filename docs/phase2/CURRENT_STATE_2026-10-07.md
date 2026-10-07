# Phase II current state — 2026-10-07

Status: **canonical current-state entry point**

This document supersedes older architecture summaries as the place to start. Historical documents remain valid records of what was believed or tested at the time; they are not deleted or rewritten retroactively.

## Executive conclusion

The project has converged away from whole-model Mirrorization.

The current architecture hypothesis is:

> **Keep the Transformer backbone canonical and shared. Put only the condition-specific / expert-specific residual into a sparse compositional rule bank. For one token/context, select multiple shared rule atoms and compose them into a virtual expert. Mirror geometry is one low-description parameterization of those rule atoms, not the organizing principle for the whole model.**

Current evidence is synthetic and small-scale. There is no natural-language LLM capacity claim.

## Current architecture

```text
token / context
    |
shared embedding + attention
    |
shared FFN base
    |
    +------------------------------+
    | sparse compositional residual|
    | router -> top-k rule atoms   |
    | R3 + R8 + R17 + ...         |
    +------------------------------+
    |
output
```

Canonical specification: [ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md](ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)

## What is currently supported

### Fact

- Whole-model Mirrorization is not required for the strongest observed effects.
- Mirror count alone did not produce a stable capacity advantage in earlier experiments.
- Shared sensor/world cores transferred changed laws better when sensor identity was kept out of the shared core.
- Residual Views were useful only when calibration left meaningful condition-specific mismatch.
- In SRM001, sparse **multi-rule** composition was much stronger than top-1/top-2 selection on the decomposable synthetic task.
- At private=32 in SRM001, shared low-rank top-4 reached about 99.9% held-out at 500 updates versus about 66.9% for a byte-near standard full-FFN MoE.
- A structured Mirror residual preserved the same qualitative signal at much lower storage than one-full-expert-per-rule controls.
- LM-loss-only sparse routing worked on the tested synthetic task and beat the near-byte standard MoE in 4/4 private=32 worlds.
- Standard MoE improves substantially with more training; near-convergence capacity superiority is therefore **not yet established**.
- A parity adversary failed for both shared/Mirror and standard variants. Current composition is not universal.

### Interpretation

The current strongest hypothesis is not "Mirror creates more capacity by itself."

It is:

> **If specialist behavior is factorable into reusable functional rules, storing one shared backbone plus a sparse bank of reusable rule coordinates may be more storage- and learning-efficient than storing many independent full experts.**

### Unresolved

- true near-convergence capacity frontier;
- ordered / non-commutative rule composition;
- routing discovery without explicit rule-like tokens;
- active-compute and wall-clock frontier;
- transfer to natural-language Transformers;
- whether Mirror geometry is better than simpler low-rank rule parameterizations after all budgets are matched.

## Evidence line

| Line | Role | Current status |
|---|---|---|
| Phase I / FQC | post-training functional compression | useful negative controls; no general sharing win |
| MN007–MN010 | native shared-state / Mirror exploration | factor reuse trainable; count alone weak; breadth matters |
| MT / RF / reachability | routing, view, geometry diagnostics | whole-view capacity claim failed; robustness / analysis signals only |
| MS008–MS014 | shared world-core + residual View decomposition | shared-core isolation and residual specialization supported; quality-equivalent compression not reached |
| SRM001 | sparse compositional shared-rule MoE | strongest current signal; multi-rule composition and LM-loss-only routing promising |

Detailed registry: [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)

## Decision rules

A claim of capacity improvement requires:
1. actual serialized bytes;
2. a controlled compute frontier;
3. explicit rule-retention / quality gates;
4. near-convergence checks, not a single update count;
5. strong Dense / standard MoE / low-rank controls.

A claim of Mirror-specific value additionally requires Mirror to beat a non-Mirror shared-rule parameterization at comparable bytes and compute.

## Current next gate

The next decisive experiment is ordered non-commutative composition:

```text
A(B(x)) != B(A(x))
```

Compare:
- sequential independent full experts;
- sequential shared low-rank rule atoms;
- sequential Mirror rule atoms;

under actual serialized-byte and training-compute frontiers.

Only after that should the project spend significant budget on natural-language scaling.
