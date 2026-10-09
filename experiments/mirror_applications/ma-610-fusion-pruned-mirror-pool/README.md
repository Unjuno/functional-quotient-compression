# MA-610 — Fusion-guided pruning of a Mirror adapter pool

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `84687450` (worker-ready baseline plus MA-602–608 evidence)
Prior art: PA125 AdapterFusion; PA126 AdapterDrop

## Hypothesis

H: In an overcomplete pool of eight shared-basis logical adapters, fusion usage identifies four useful Views whose removal preserves held-out composition quality better than random pruning, while compacting actual serialized bytes and matching a four-view model trained from the start.

## Mirror insertion

> **Mirror insertion:** this experiment adds one learned Givens angle per logical adapter to shared rank-2 factors, then uses learned fusion usage to select and serialize a compact subset of four Views instead of retaining eight physical adapter codes.

Controls: independent full adapters with the same fusion gate; random four-view pruning; and four-view-from-start training. Pruning decisions use only training-set fusion usage; no post-prune fine-tuning.

## Gates

PASS requires usage-pruned Mirror held-out normalized MSE <=1.10× the four-view-from-start Mirror, <=1.10× full overcomplete adapter bank, and <=0.50× mean random-prune error in both development worlds; payload <=0.75× overcomplete Mirror. Fresh replication runs only if all development gates pass.

## H / T / D / C / U

**H — hypothesis:** fusion usage can identify four useful Views in an overcomplete eight-view Mirror adapter pool, allowing pruning that retains quality better than random pruning and matches four-view-from-start training.

**T — execution:** CPU PyTorch 2.14.1; synthetic 12→rank-2→10 adapter mixture with four target functions and eight learned slots. Two development worlds 61001/61002; 4,096 train and 1,024 held-out examples; 1,200 updates for overcomplete and small-from-start Mirror/full models. Usage ranking used training-set mean router probabilities only. No post-prune fine-tuning. Ten random four-slot masks were evaluated per model/world. Fresh 61011/61012 stayed sealed.

**D — FAIL:** Mirror usage-pruned NRMSE2 was 0.00510/0.00890, versus full overcomplete Mirror 0.0000436/0.0000777 and four-view-from-start Mirror 0.0000660/0.00000529. It beat mean random-prune error (0.0935/0.1632) by about 18×, but missed both quality comparators by large margins. More critically, actual usage-pruned payload was 3,285 B versus 3,225 B overcomplete and 3,033 B small-from-start; the retained-index metadata erased any storage saving. Pruned compute proxy dropped from 480 to 240 MAC/example, but the quality/byte gates failed.

**C — strongest counter-hypothesis:** average fusion probability is a poor proxy for functional importance. A lower-usage View may still carry a behavior needed for a subset of inputs; pruning without fine-tuning removes that behavior. A model trained from the start with four slots can reorganize its basis and therefore outperforms a post-hoc subset.

**U — boundaries:** synthetic teacher, fixed update budget, and one pruning rule. No natural source tasks, adaptive post-prune distillation, or language models tested.

## Facts / interpretation / hypothesis

- **Fact:** 16/16 saved payloads match exact byte sizes/hashes and reload exactly; ten random masks were measured per mode/world; two tests pass.
- **Interpretation:** fusion weights rank candidates better than random selection here, but that ranking alone does not preserve the full pool's function or produce a smaller inference payload.
- **Hypothesis:** a useful compact pool likely needs post-prune recovery/distillation and a selection score that captures conditional task coverage, not only average router mass.
