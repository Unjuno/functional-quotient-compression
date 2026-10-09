# MA-517 status

- Status: FAIL
- Branch: `research/ma-517-function-vector-composition-20261009`
- Base: `64f4d3b0`
- Model: pinned GPT-2 revision used by MA-516, CPU float32
- Development: a smoke check validates prompts, target tokenization, and vector extraction; no condition selection
- Fresh: completed, worlds 51710-51712 × seeds 0-2
- Result: 54 rows; every method 0% exact accuracy; direct ICL also 0%; mean vector cosine 0.99968; two-vector payload 99,945 B
- Interpretation: invalid/ineffective task and extraction setup prevents a composition claim; negative result retained

H: Role-specific layer placement for two extracted function vectors improves held-out composition over raw addition.

T: Fruit→color plus color→shape composition, with query-only/direct ICL, raw sum, factorized layers and single-vector ablations.

D: FAIL. Direct ICL failed its minimum viability gate in every world, and factorized vectors did not beat raw-sum accuracy (both 0%).

C: Native direct ICL also failed, so prompt/task serialization failure is a stronger explanation than a Mirror-specific failure. Near-identical vectors point to a non-discriminative extraction contrast.

U: A validated compositional prompt and causal extraction method, natural tasks, and any learned code remain untested.
