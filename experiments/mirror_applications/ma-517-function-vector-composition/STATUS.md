# MA-517 status

- Status: SCREENING
- Branch: `research/ma-517-function-vector-composition-20261009`
- Base: `64f4d3b0`
- Model: pinned GPT-2 revision used by MA-516, CPU float32
- Development: a smoke check validates prompts, target tokenization, and vector extraction; no condition selection
- Fresh: worlds 51710-51712 × seeds 0-2, locked until protocol/source freeze

H: Role-specific layer placement for two extracted function vectors improves held-out composition over raw addition.

T: Fruit→color plus color→shape composition, with query-only/direct ICL, raw sum, factorized layers and single-vector ablations.

D: pending.

C: The underlying extracted vectors may not encode reusable functions, as MA-516 showed.

U: All fresh metrics.
