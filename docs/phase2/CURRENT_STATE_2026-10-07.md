# Phase II current state — 2026-10-07, through SRM003

Status: **current navigation and interpretation, not a preregistration**.

## Executive conclusion

The hypothesis is a canonical shared backbone plus multiple reusable specialist residuals and only-when-needed private capacity. Whole-model Mirrorization, ES and recurrent training loops are not the default.

**SRM003 was executed, but the scientific adoption gate failed.** A small causal decoder learned all atomic mappings but did not reliably compose them from endpoint-token loss. More residual experts and validation-based pruning did not resolve this. The next bottleneck is the interface and execution structure connecting learned operators, not merely the number of stored atoms.

## What the latest evidence actually says

### SRM001: decomposable synthetic tasks

Multiple selected components had a strong fixed-update advantage in the count task and weaker signals on bitmask tasks. LM-loss-only routing still received explicitly structured rule-token inputs. The result is not unconstrained rule discovery in language. Some reported retention counters aggregate compound examples containing a rule; they must not be substituted for exhaustive atomic truth-table retention. Standard MoE improved with longer training, so near-convergence capacity superiority was not established.

### SRM002: ordered operators with a supplied executor

A teacher generated from a shared signed basis plus sparse rank-2 private residuals could be learned and compacted efficiently. Mirror stretch/shear was weaker than a signed shared basis on that family. Learned residual scores separated teacher-private slots in that controlled setting.

The operator fixture explicitly passed the updated state through the supplied ordered list of operators. It demonstrated learning reusable operators **with an executor**, not discovering an executor from a conventional decoder's final loss. Failure of a narrow shared model on another discrete family does not prove that family is intrinsically unshareable.

### SRM003: no oracle router or private mask

Two-layer causal Transformer, width32, 24 operators on16 states, explicit operator tokens, final answer CE. Three fresh world/init pairs; no auxiliary intermediate labels. Models train on both atomic and two-rule examples; pair combinations are held out with reverse-closed splits.

- All five main architectures retained all24 atomic rules over every one of16 inputs at1600 updates.
- Unseen-pair accuracy was only about9–16% across the five methods at1600 updates.
- Dense, full-MoE and Hybrid long controls reached about13–19% at6400 updates; they still retained all atomic rules.
- Hybrid did not satisfy the predeclared2-point advantage over both full-MoE and low-rank-MoE in every fresh setting.
- Two externally orchestrated calls using the model's **predicted**, not true, intermediate state scored100% in all9 long models and their1600-update parents. This supplies execution order and costs extra compute; it is not a one-forward success.
- Validation-pruning reduced Hybrid inference bytes90,329 ->85,672 (5.16%). It passed a numerical1-point tolerance at low composition accuracy, but was not consistently better than random pruning or small-from-start.
- Removing atomic examples gave zero exactly retained atomic rules in the pair-only diagnostic; query-position exposure also changed, so this is not a single-factor causal proof.

Primary comparisons controlled data/update opportunities, not total FLOPs or wall time. No LLM or general capacity claim is made.

## Interpretation and open hypotheses

Fact: reusable atomic mappings can exist without successful in-model ordered execution.

Interpretation: shared/private FFN parameterization alone does not impose the state-passing structure needed for non-commutative composition. The mechanism could involve optimization, representation of intermediate states, placement, routing or limited depth; the present diagnostic does not uniquely identify one cause.

Hypothesis: a fixed-depth, non-recurrent composition interface may allow the learned rules to be reused internally. This has NOT been tested by SRM003. Do not label it an adopted solution or promise a gain.

## Active evidence map

| Line | Role | Scope/status |
|---|---|---|
| Phase I/FQC | post-training compression | preserved; no general sharing win |
| MN/MT/RF/reachability | views, geometry and routing | count-only weak; robustness and analytical diagnostics |
| MS008–MS014 | sensor/world decomposition | functional transfer signals; quality-equivalent compression not reached |
| SRM001 | shared-rule synthetic causal tasks | controlled fixed-update signals; parity failed |
| SRM002 | operator composition and shared/private pruning | controlled factorized teacher with supplied execution |
| SRM003 | causal discovery and pruning without oracle routing | completed negative gate; storage and execution separated |

## Decision rules

Keep learning efficiency, observed retention, mathematical capacity, rule reuse and compression separate. Require actual serialized bytes, strong Dense/top-k MoE/low-rank controls, and compute frontiers for any performance claim. A longer unsuccessful training run is not a certified upper capacity bound. Mirror-specific value additionally needs a non-Mirror shared-rule comparison.

Primary scientific adoption gate: **FAIL**. Pruning hardware/byte mechanics: verified. Useful-quality compression: **not established**. Natural-language evidence: **not tested**.

## Navigation

- [SRM003 report](SRM003_CAUSAL_DISCOVERY.md)
- [SRM003 runnable source, tests and counts](../../experiments/shared_rule_moe/srm003_20261007/README.md)
- [SRM002 original record](SRM002_NONCOMMUTATIVE_COMPOSITION.md)
- [SRM001 original record](SRM001_SHARED_RULE_MOE.md)
- [Architecture hypothesis](ARCHITECTURE_SPARSE_SHARED_RULE_MOE.md)
- [Experiment registry](EXPERIMENT_REGISTRY.md)
- [Roadmap](../../roadmap/PHASE2_SPARSE_RULE_ROADMAP.md)

Historical records are preserved. This document corrects earlier overbroad interpretations without replacing their original measurements or protocols.
