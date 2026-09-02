# Root Selection and Pricing: D64–D69

D64–D69 extend shared-atom allocation from a fixed set of roots to the harder problem of discovering, opening, replacing, and certifying roots/dictionaries.

## Two separate proof obligations

1. **Master optimality:** given a declared root pool, find the best block assignment and paid-atom union under the budget.
2. **Candidate-family completeness:** establish that no omitted root or root coalition can improve the result.

An exact master solves only the first problem.

## D65: one-column pricing is not a stopping certificate

The historical counterexample has a 10-bit budget and two private 5-bit blocks. Roots A and B share a 4-bit prerequisite Q; each root costs 1 bit and each served-block assignment costs 1 bit.

- private plan: 10 bits;
- A alone plus the other private block: 11 bits, infeasible;
- B alone: 11 bits, infeasible;
- A+B together: 8 bits and zero error.

Thus no feasible improving single omitted root does **not** imply that no improving omitted coalition exists. Shared prerequisite amortization creates complementarity.

## Finite-family certificate

For a finite declared root family R and hard root-count cap K,

`min_{S subset R, |S| <= K} ExactMaster(S)`

is the exact family-relative optimum. The number of root sets is

`sum_{k=0}^K C(|R|, k)`.

D65 used 100 roots with K=2, hence 5051 root sets. This is exact relative to that declared family, not a proof that the root generator is complete for a real model.

## D67–D69: replacement and family bounds

Later deltas show that augment-only pricing can also stop at a bad incumbent: dropping an active root and opening a complementary pair can improve the reduced objective even when no single addition or single replacement does. Safe family/group lower bounds can prune many coalitions, but a negative lower bound is inconclusive and requires tighter or exact evaluation.

## Canonical optimizer rule

The FQC optimizer should maintain separate status for:

- `MASTER_SOLVED`: optimal among the active/declared candidate set;
- `FAMILY_CERTIFIED`: all legal root coalitions up to the declared cap have been priced or safely bounded;
- `GENERATOR_OPEN`: the root proposal family itself is heuristic/incomplete.

A real-model result must not present `MASTER_SOLVED` as global root optimality unless the latter obligations are also satisfied or explicitly scoped away.
