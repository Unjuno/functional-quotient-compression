# MA-466 status

- Status: FAIL for Mirror-specific advantage; structured factor composition PROMISING in this synthetic family
- Branch: `research/ma-466-unipelt-mirror-axes-20261009`
- Base commit: `e8bdd67d`
- Protocol freeze: `f2255673`
- Fresh worlds: 46610–46612 × seeds 0–2
- Results/registry/ledger: pending final commit

## H / T / D / C / U

- H: Factorized Mirror axes combine UniPELT mechanisms compactly across held-out task/layer/position combinations.
- T: Three basis functions (LoRA, prefix, adapter) with rank-one product gates; held-out factor combinations, CP, MLP, per-combo gate and ablation controls.
- D (Fact): N36 Mirror/CP NRMSE 0.000203 at 2,653B, exact equal output/hash; HyperFormer 2.438 at 4,569B; per-combo support-fit gates exact at 2,469B. Ablating each component raises error.
- D (Interpretation): Factor composition is effective in this deliberately structured family, but ordinary CP is exactly equivalent and support-fit gates are smaller/more accurate.
- C: Teacher construction is exactly rank-one CP across factors; generic tensor factorization explains Mirror behavior.
- U: Natural UniPELT fine-tuning and higher-rank interactions remain untested.

## Next action

Commit evidence, update registry and claim ledger, push branch, then continue with MA-468.

## Blockers

None.
