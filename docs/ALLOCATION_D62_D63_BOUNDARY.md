# Allocation Boundary: D62 vs D63

D62 solves the sensitivity-weighted hard-budget allocation problem when the variable codec state is block-separable after mandatory fixed overhead has been charged.

For block `i`, choose one option with deterministic bits and certified weighted error. The exact problem is a multiple-choice knapsack. Dominated block options can be removed, exhaustive/DP solution gives an integral upper witness, and the Lagrangian dual gives a lower bound on the optimum.

The fixed overhead must be removed first:

`B_var = B_total - B_fixed`.

If `B_fixed > B_total`, the rate target is impossible before local allocation.

## Why D62 is not the full FQC allocator

D62 assumes the cost of choosing one block option is separable from the choices of other blocks. This assumption fails when an option conditionally opens a shared root, dictionary, basis, selector table, or other PAID prerequisite.

Once such conditional shared state exists, candidate rate becomes

`private payload + cost(union of shared prerequisite closures)`

and D63/shared-allocation logic is required.

Therefore:

- use D62 for independent block payload after all mandatory shared state is fixed;
- use D63 for conditional shared paid atoms;
- do not fold a shared root's fixed cost into every consumer, and do not ignore it either.

## Greedy warning

The archived D62 counterexample uses a 10-bit budget and indivisible upgrades A=(6 bits, gain 10), B=(5,8), C=(5,8). Density greedy selects A for gain 10, while the exact choice B+C uses 10 bits for gain 16. Marginal density is not an exact discrete allocation rule without additional convexity/polymatroid structure.
