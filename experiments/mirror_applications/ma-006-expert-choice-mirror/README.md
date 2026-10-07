# MA-006 — capacity-balanced expert-choice Mirror

Status: **PROMISING** for the registered expert-choice sharing gate; expert-choice is dominated by token-choice on this task
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME
Branch: `research/ma-006-expert-choice-mirror-20261007`
Base commit: `eca36e2` (verified MA-004 parent)

## H — hypothesis

Under expert-choice routing with a fixed token capacity per expert, one nonlinear FFN plus per-expert Mirror views can retain the routed quality of a full expert bank while reducing serialized bytes. The effect must beat ordinary expert tying and byte-near FiLM/rank-2 residual controls; standard token-choice routing is a route-policy comparator.

## Prior-art delta

MA-001/002/004 studied token-choice top-1/top-2/dense-soft routing. This experiment fixes how many tokens each expert may select, allowing overlap and unassigned tokens. The score router chooses tokens per expert; each expert has capacity exactly batch_size/4. PA01 hard tying and PA03 low-rank routers are included; PA02 path constraints are outside this single-layer task.

## Physical-to-logical claim

One 16→32→16 GELU FFN is shared across four expert-choice slots. Each slot chooses a fixed-capacity set, then uses its expert-specific view. A token selected by multiple experts receives a normalized weighted output; an unselected token receives zero on the MoE branch. This explicit no-route behavior is charged in quality, not repaired by an oracle fallback.

## T — protocol

See `PROTOCOL.json`. Each training batch has 16 examples per role; role labels are supervised router targets and inputs have overlapping Gaussian class clouds. The aligned teacher uses one FFN under Givens views; independent mode uses four unrelated FFNs. Compare full expert-choice, dense token-choice, rank-2 router variants, hard tying, FiLM, rank-2 residual, and Mirror. Development chooses LR; fresh worlds are gated and frozen.

## Storage and compute

Count actual serialized inference payload bytes including router, full/shared experts, views/control tensors and metadata. Record capacity, expert loads, token coverage, role recall, active FFN assignments, coordinate FLOPs, wall time and CPU throughput.

## Results

Development selected LR 0.003 by the registered mean-MSE rule. The aligned Mirror/full expert-choice MSE ratio was 1.005 and payload ratio 0.333, passing the development gate. Fresh worlds 60001–60003 ran at the frozen setting; all 48 deterministic fresh result rows replayed exactly.

### Fact

- Aligned expert-choice Mirror MSE was 0.01571 / 0.00820 / 0.01156, or 0.975x / 0.954x / 1.006x full dense expert-choice MoE. Coverage was 0.807–0.824 vs 0.809–0.818 full expert-choice; role recall was 0.757–0.774 vs 0.760–0.773. Each expert received exactly 512 slots on validation (capacity enforced). The registered quality/coverage/bytes gate passed 3/3.
- Mirror payload was 7,761B vs 23,341B full expert-choice MoE (0.333x); hard-tied expert-choice used 7,509B, FiLM and rank-2 residual 8,909B. Mirror MSE was 0.805–0.878x tied, 0.877–0.927x FiLM, and 0.899–0.934x residual. The Mirror-specific gate failed: 10% quality margin was not reached against FiLM/residual in every world, and hard tying is 252B smaller with lower active compute.
- The strongest route control, dense token-choice top-1, covered 100% of tokens and had 2.50–3.42x lower MSE than Mirror at similar full-model bytes. Expert-choice deliberately allows overlap and leaves 18–19% of tokens with no MoE assignment in these runs; no fallback was supplied.
- Independent expert functions were not recovered by one shared view bank. Mirror MSE was 0.0275–0.0287 vs 0.0191–0.0211 for full expert-choice and 0.0159–0.0179 for token-choice.
- Median aligned training wall was 8.47s Mirror vs 1.45s tied; median inference throughput was 0.549M vs 2.006M examples/s for tied (0.274x). Mirror was 0.412x full expert-choice throughput. This eager CPU result is implementation-specific.
- Three tests passed. Serialization/load output equality was checked on all 96 runs; all 48 fresh deterministic rows replayed exactly, excluding timing fields.

### T — execution

Eight methods (full expert-choice, full token-choice, rank-2 router expert-choice/token-choice, hard tying, FiLM, rank-2 residual, Mirror) trained for 1,200 updates × 64 examples in aligned and independent modes. Each batch contains 16 examples per role. Each expert independently selects its highest-scoring 16 tokens; overlaps are permitted and unassigned tokens receive zero on the MoE branch. Development world 60000 selected LR 0.003. Fresh worlds 60001–60003 used locked seeds and configuration. Actual serialized payloads include all learned state and metadata.

### D — decision: PROMISING

The registered Mirror expert-choice quality/storage gate passed in 3/3 aligned worlds, with 66.7% fewer payload bytes than the full expert bank and quality near full expert-choice. The experiment does not show that expert-choice itself is a better routing policy here: standard token-choice improves coverage to 100% and MSE by 2.5–3.4x. Mirror gains are also modest against FiLM/residual and incur substantial eager runtime cost. The result maps a narrow shareable-expert case and a separate routing-policy limitation.

### C — strongest counter-hypothesis

Balanced synthetic class clouds make router supervision clean, but the best token-choice result suggests expert-choice's overlap/no-route pattern, rather than expert representation, drives much of the quality loss. A token fallback or residual path could recover quality, but that is outside this frozen candidate and would require a new experiment/amendment.

### U — unconfirmed

Natural-language expert-choice quality, alternative no-route handling, larger batch/capacity factors, near-convergence/fixed-byte capacity, optimized dispatch kernels, and generic byte-matched shared-basis controls remain untested.

## Fact / interpretation / hypothesis

- **Fact:** measured outputs and gates are in `RESULTS_CORE.csv`; route loads are explicitly reported and actual serialized bytes are authoritative.
- **Interpretation:** expert-choice can support a shared-view expert bank at reduced bytes on related functions, but no-route coverage and token-choice's stronger quality limit its usefulness in this task.
- **Hypothesis:** Mirror views could be useful in large expert-choice systems with a residual or fallback path; that is not established here.
