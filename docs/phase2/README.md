# Phase II — Mirror-native conditional models

**Active, unfinished research.** One shared FFN per layer is modulated by a small state bank and a token-local router, all trained together. The model does not store a separate full FFN for every state or materialize one for every token. The current implementation is feature-wise modulation, not a literal frequency/mirror transform or sparse-MoE kernel.

## Evidence map

| Experiment | Role | Status |
|---|---|---|
| MN001 | Initial 2-layer causal Transformer / 4-rule toy | Learned, routing ablation and cache feasibility; ordinary Dense also solved it. Historical local package preserved. |
| [MN002](../../experiments/mirror_native/mn002_20261003/README.md) | 16-rule x 32-symbol task, 24 fixed-budget training runs | Mirror4 wins against matched Dense in 3/3 seeds; Mirror16 loses in 3/3. |
| MN002R | New task table and 9 fresh training runs | Mirror4 wins Dense in 2/3, simple gate in only 1/3; superiority not established. |

## What is currently useful

The inspectable implementation demonstrates learned state-dependent computation without token-specific full-matrix expansion; cache equivalence and actual model-file accounting are tested. It supplies reproducible controls and failure conditions, not a capacity multiplier or an LLM performance claim.

Published scopes must remain separate: **numerical identity**, **successful learning**, **router usage**, **equal-resource predictive advantage**, **scaling**. The first three have small-test evidence. The last two are unresolved. A router ablation is not a proof that a retrained static model would fail.

## Roadmap

1. Preserve the working model, protocols, all seed results and failures.
2. Evaluate non-saturating tasks and multiple independent task distributions.
3. Match parameter bytes and training/inference computation against Dense, direct gates, FiLM-style modulation and shared-expert baselines.
4. Only then seek larger-scale independent replication. No paid compute is required by the current prototype.

Phase I remains [preserved and paused](../phase1/STATUS.md), not deleted or scientifically declared complete. Existing Apache-2.0 terms are unchanged. Use the experiment's `CITATION.cff`; citation is requested, not an extra legal condition.
