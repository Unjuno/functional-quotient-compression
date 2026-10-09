# MA-498 status

- Status: FAIL
- Branch: `research/ma-498-code-distance-regularizer-20261009`
- Base: `eb5c7a1b`; A1 excludes duplicate-code A0
- Fresh: 49820-49822 × seeds 0-2

H: Max-min distance selection improves noisy routing at equal 8-bit budget.

T: Random unique vs development-selected 8-bit IDs and ordinary binary IDs under bit flips.

D: FAIL. At p=.1, selected and random both accuracy .667, minimum distance 1, payload 2,085B. Binary accuracy .595. Extra address bits help; selected distance regularization adds no benefit over random.

C: No improvement in minimum distance was available at this 8-bit/32-ID code pool.

U: Structured ECC and task-level router errors.
