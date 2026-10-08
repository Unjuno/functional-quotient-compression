# MA-008 — hierarchical Mirror-MoE

Status: **PROMISING** for aligned expert sharing; hierarchy itself had no consistent quality advantage over flat routing.
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME.
Branch: `research/ma-008-hierarchical-mirror-moe-20261007`.
Base commit: `993c4ce` (verified MA-007 parent).

## H — hypothesis

For four related nonlinear expert roles arranged as two groups of two, a hierarchical group→expert router plus one shared FFN and role-specific Givens views can recover untied hierarchical MoE quality with fewer serialized bytes. It must beat hard tying and byte-near FiLM/rank-2 residual controls; flat and factorized routing test whether the hierarchy adds value.

## Prior-art delta

MA-001/002/004/006/007 measured flat token-choice, sparse top-k, dense-soft and expert-choice variants. This experiment adds two-level group/child routing over balanced roles. PA02 concerns shared/path routing across blocks; PA03 low-rank routing is represented by the rank-2 flat full-MoE control.

## T — execution

Seven methods (full flat, full hierarchical, rank-2 flat, tied hierarchical, FiLM, rank-2 residual, Mirror hierarchical) trained 1,200 updates × 64 examples in aligned and independent teacher modes. Each batch has 16 examples per role. Development world 80000 selected LR 0.003 using mean MSE across methods and modes. Fresh worlds 80001–80003 used frozen seeds and settings. Hierarchical routes first select one of two groups and then one of two child experts.

## Results

### Fact

- Aligned Mirror MSE was 0.004516 / 0.006234 / 0.004224, or 0.948x / 0.949x / 0.931x full hierarchical MoE. Role accuracy was within 0.15 percentage points of the full hierarchical control. The registered quality/storage gate passed 3/3.
- Mirror payload was 8,898B vs 24,673B full hierarchical MoE (0.361x; 63.9% fewer bytes), and 23,341B full flat MoE (0.381x). Hard-tied hierarchy used 8,581B, 317B fewer. FiLM and rank-2 residual each used 10,111B.
- Mirror MSE was 0.580–0.667x hard tying, 0.688–0.774x FiLM, and 0.739–0.796x rank-2 residual. The Mirror-specific byte condition failed vs hard tying, despite lower MSE. Mirror active proxy was 1,088 MACs plus 48 coordinate FLOPs per example; tied was 1,088 MACs.
- Full hierarchical routing itself did not consistently beat flat full routing: hierarchical/flat MSE ratios were 0.960 / 1.028 / 1.026 across worlds. Mirror hierarchy did beat flat full MoE MSE in all three (ratios 0.910 / 0.976 / 0.955), while using fewer bytes.
- Independent roles were not fully recovered: Mirror MSE was 0.02197–0.02652 vs 0.01489–0.01921 for full hierarchical MoE. This remains a private/richer-state boundary.
- Median aligned training wall was 9.79s Mirror vs 2.24s tied; median CPU inference throughput was 0.653M vs 2.056M examples/s (0.317x tied). Eager CPU timing is implementation-specific.
- Three tests passed. Exact serialized round-trip was checked across 84 runs; all 42 deterministic fresh rows replayed exactly, excluding timing fields.

### D — decision: PROMISING

The registered aligned useful-sharing gate passed in 3/3 worlds, with 63.9% fewer bytes than the untied hierarchical model. Mirror improves the quality of hard tying, FiLM and residual controls. It is not a strict Pareto improvement over hard tying because that payload is 317B smaller and the eager view implementation is slower. The hierarchical router is not established as a quality improvement over flat routing in this screen.

### C — strongest counter-hypothesis

The teacher is deliberately generated from Givens views, which favors Mirror. Balanced noisy role labels make the group/child hierarchy unusually clean. A generic byte-matched shared basis may close the quality gap without Mirror coordinates.

### U — unconfirmed

Natural-language MoE quality, near-convergence fixed-byte capacity, larger hierarchies, generic generated-router/basis controls, optimized GPU kernels, and the full private-parameter boundary remain untested.

## Fact / interpretation / hypothesis

- **Fact:** all values and route metrics are in `RESULTS_CORE.csv`; storage uses actual serialized payload bytes.
- **Interpretation:** shared Givens views recovered related nonlinear roles at substantially lower storage; the hierarchy itself offered no consistent benefit, and eager Mirror runtime regressed.
- **Hypothesis:** trained experts may form useful low-description groups; this experiment does not establish that in language models.
