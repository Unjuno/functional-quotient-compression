# MA-189 status

- Status: SCREENING
- Branch: `research/ma-189-freeze-backbone-mirror-20261007`
- Base commit: `8d0763ddc241667e6e5c0fdefad16b2d71152538`
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## H — hypothesis

Two task-specific Givens coordinates on a frozen shared base will match rank-2 LoRA at 64 examples with <=0.25x incremental inference bytes on the matching two-sided task family; independent tasks will require private parameters.

## T — fixed setup

16D-to-8D synthetic map; one held-out skill per world; budgets 16/64/256 examples and 200 updates each. Development seeds 18901–18902 choose LR .003/.01. Fresh seeds 18911–18913 remain sealed. Controls: hard tie, one-sided Mirror, rank-1/2 LoRA, non-degenerate shared rank-2 hypernetwork, independent full map.

## D — development screen

The registered all-method/all-condition mean MSE at 64 examples selected LR 0.01 (0.6240 vs 0.6548 for LR 0.003). At 64 aligned examples, two-sided Mirror fit the view-aligned task near numerical precision; on unrelated task matrices it remained poor and rank-2 LoRA was better. Development did not meet the stop rule that both worlds miss the 256-example quality or byte gate, so fresh worlds remain authorized by the preregistered plan.

## C — strongest counter-hypothesis

Two angles may be an arbitrary low-dimensional task code with no advantage over rank-1 LoRA or a compact hypernetwork once metadata and compute are charged.

## U — unresolved

No result yet. The synthetic aligned family does not imply natural-task or language-model utility.
