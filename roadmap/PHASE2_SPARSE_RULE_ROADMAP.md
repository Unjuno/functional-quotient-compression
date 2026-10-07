# Phase II roadmap — shared rules, private residuals and execution

Date: 2026-10-07, after SRM003 and TM001.

Systematic application exploration is tracked separately in [MIRROR_APPLICATION_ROADMAP.md](MIRROR_APPLICATION_ROADMAP.md); this file remains the shared-rule/execution roadmap.

## R0 — Repository/state consolidation

Current entry points, historical records and SRM experiment paths are separated. SRM003 adds complete executable code, tests, original plan, frozen configuration and exact fresh counts. Main and historical experiments are not overwritten by this research branch. Preserve source hashes and negative outcomes.

## R1 — Ordered composition mechanism

SRM002 completed an operator-level mechanism probe with explicit ordered execution. A signed shared basis and sparse private corrections learned the factorized teacher efficiently; the tested Mirror family did not. This gate does not establish automatic execution discovery by a causal LM.

SRM003 subsequently removed oracle routing/private masks in a small causal Transformer. Atomic retention reached100%, but direct unseen ordered composition remained poor even at6400 updates. The same models reached100% only when called twice with a predicted intermediate state and an externally supplied procedure. Execution completion: DONE. Scientific automatic-composition/adoption gate: FAIL.

## R2 — Capacity versus learning efficiency

Still open. Require byte/compute/quality frontiers and adequate task accuracy. Neither a favorable fixed-update snapshot nor an unsuccessful longer run establishes a capacity upper bound. Do not widen banks solely because composition failed while atomic mappings were already retained.

## R3 — Next bounded hypothesis: learned intermediate-state interface

PROPOSED, NOT EXECUTED: test a fixed-depth feed-forward architecture that passes the output of one learned rule to the next, without true intermediate targets at training or test time. Keep ordinary Dense, top-k MoE and shared-rule baselines and the two-forward external-executor control. Explicitly state which execution structure is supplied and which is learned. No ES or recurrent training loop is assumed.

Before making a performance claim, separate atomic learning, intermediate-state transport, routing, final readout, sequence-length transfer and active compute. Use a development-only protocol and fresh worlds. A human-readable reasoning trace is not required, and an internal state should not be assumed to have a particular semantic interpretation without intervention tests.

## R3b — TM001 temporal packetization result

TM001 tested P learned future slots in one forward. When all information determining the packet was available at macro-step start, P=4 replicated at essentially 100% joint accuracy and P=8 worked on one fresh world. A one-thread CPU benchmark showed low-batch throughput gains versus KV-cached AR. When one packet-level branch variable was hidden, factorized slots and triangular latent mixing failed to preserve the joint trajectory; a preliminary shared packet latent improved but did not close the gap.

Next bounded hypothesis: combine a fixed-depth intermediate-state / packet-latent interface with parallel phase slots. Do not scale P or claim language speedup until joint consistency and GPU behavior are measured.

## R4 — Atom family competition

Only after useful composition accuracy: compare signed low-rank basis, independent low-rank residual, standard full experts, and Mirror controls at actual bytes and measured compute. Mirror survives only when it improves the relevant frontier. Do not claim arbitrary rules are unshareable from failure of one narrow parameterization.

## R5 — Learn-many, prune and compact

SRM003 physically removed half the additional residual slots, saving5.16% of model bytes, but did not show consistent selected-pruning benefit over random pruning or small-from-start. Continue only with useful task accuracy and held-out deletion audits. Top-k activation can stay constant after pruning, so storage reduction is not automatically runtime reduction.

## R6 — Natural-language external validity

No natural-language capacity claim follows from SRM001–003. A future tiny-LM comparison may test external validity once a clearly scoped mechanism or practical frontier is established. Natural-language NLL is a quality measure, not a ground-truth count of stored independent rules. Compare Dense, standard sparse MoE, low-rank MoE and any justified shared-rule variant; record bytes, data, full training cost and inference throughput.

## Stop and pivot criteria

- Preserve failures and report whether the limit is observed optimization or demonstrated representation.
- Remove added machinery if a simpler control matches its quality/cost frontier.
- Keep the executor-supplied and executor-learned evidence lanes distinct.
- Do not change an audit split into a tuning set or label a rewritten retrospective plan as preregistration.

[Latest current state](../docs/phase2/CURRENT_STATE_2026-10-07.md) · [SRM003 report](../docs/phase2/SRM003_CAUSAL_DISCOVERY.md)
