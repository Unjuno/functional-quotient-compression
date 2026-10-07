# MA-007 — token-choice Mirror routing crossover

Status: **PROMISING** for token-choice sharing; the hard-tied model is smaller, while expert-choice loses coverage.
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME.
Branch: `research/ma-007-token-choice-mirror-20261007`.
Base commit: `a556e63` (verified MA-006 parent).

## H — hypothesis

With one top-1 token-choice assignment per input, a shared nonlinear FFN plus per-role Givens views can recover full-model quality while retaining every token and using fewer serialized bytes. On matched role data, token-choice Mirror should reduce the no-route loss observed for expert-choice Mirror in MA-006. It must beat tied/FiLM/residual controls for Mirror-specific value.

## Prior evidence delta

MA-001 tested nonlinear top-1 Givens experts on sign-quadrant inputs and passed aligned quality/storage. MA-006 found expert-choice Mirror coverage around 81%, while token-choice full MoE covered all tokens and had 2.5–3.4x lower MSE. This experiment tests both Mirror routing policies on balanced, overlapping role clouds and compares token-choice against untied, tied, FiLM, and residual controls.

## T — execution

Eight methods (full expert-choice, full token-choice, rank-2 token-choice, tied token-choice, FiLM token-choice, rank-2 residual token-choice, token-choice Mirror, and expert-choice Mirror) trained 1,200 updates × 64 examples in aligned and independent modes. Each batch has 16 examples per role. Development world 70000 selected LR 0.01 by mean MSE across methods and modes. Fresh worlds 70001–70003 used locked seeds and configuration. Router CE supervised the known synthetic role.

## Storage and compute

Actual serialized CPU state dict plus config is authoritative. All router, expert/view/control tensors and metadata are charged. Each token-choice input dispatches to exactly one role. MAC proxy, Givens coordinate FLOPs, wall time, and CPU throughput are reported separately.

## Results

### Fact

- Token-choice Mirror MSE was 0.002442 / 0.003015 / 0.002229, or 0.766x / 0.769x / 0.768x full token-choice MoE in fresh worlds 70001–70003. Coverage was 100% in every world. The registered quality/storage gate passed 3/3.
- Mirror payload was 7,761B vs 23,277B full token-choice (0.334x; 66.6% fewer bytes). Hard tying used 7,509B, only 252B less. FiLM and rank-2 residual used 8,909B.
- Mirror MSE was 0.500–0.588x hard tying, 0.586–0.659x FiLM, and 0.610–0.743x rank-2 residual. The Mirror-specific byte gate failed against hard tying: Mirror uses 3.4% more bytes despite substantially lower MSE. MAC proxy was 1,088 plus 48 coordinate FLOPs per example, 4.4% above tied MACs.
- Token-choice Mirror MSE was 0.209–0.302x expert-choice Mirror on the same fresh worlds; coverage was 1.000 vs 0.818–0.827. This is consistent with expert-choice's overlap and unassigned-token behavior limiting quality in MA-006.
- On independent role functions, token-choice Mirror MSE was 0.0214–0.0241 vs 0.0156–0.0183 for full token-choice and 0.0181–0.0215 for FiLM. A single view family did not fully reproduce arbitrary roles.
- Median aligned train wall time was 8.61s Mirror vs 1.42s tied. Median inference throughput was 0.515M vs 2.213M examples/s (0.233x tied). This eager CPU timing is implementation-specific.
- Tests: 3 passed. Serialization round-trip was checked on all 96 runs. All 48 deterministic fresh rows replayed exactly, excluding timing fields.

### D — decision: PROMISING

Token-choice Mirror retained every token and passed aligned quality/storage in 3/3 worlds, improving MSE over untied token-choice with one third the payload. The routing-policy crossover supports the MA-006 counter-hypothesis: token-choice removes the no-route coverage penalty observed for expert-choice. Mirror-specific Pareto improvement is not established because hard tying is 252B smaller and eager Mirror inference is slower. Independent functions still require private or richer state.

### C — strongest counter-hypothesis

The teacher is deliberately generated from Givens views, and the role labels are cleanly supervised. This favors Mirror. Generic byte-matched shared bases could narrow the gap; natural-language role structure is untested.

### U — unconfirmed

Natural-language MoE quality, fixed-byte capacity, optimized kernels, larger role pools, generic shared-basis controls, and the broader private-parameter boundary remain untested.

## Fact / interpretation / hypothesis

- **Fact:** all values are in `RESULTS_CORE.csv`, with serialized payload bytes and routing coverage recorded per run.
- **Interpretation:** in this aligned synthetic family, token-choice is a materially better routing policy than expert-choice for preserving quality; Mirror views can recover useful top-1 role functions at reduced bytes.
- **Hypothesis:** real MoE experts may share low-description role coordinates, but this task does not establish that for language models.
