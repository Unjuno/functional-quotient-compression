# MA-464 status

- Status: FAIL — frozen development gates failed in both seeds
- Branch: `research/ma-464-adamix-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Last verified code/protocol commit: `7ae90fc`; final result commit follows
- Development complete: yes; 10 condition/seed rows
- Fresh/audit opened: no
- Results committed: yes (this branch)
- Verification committed: yes (this branch)
- Registry row updated: yes (this branch)

## Next action

Draw the next eligible candidate uniformly at random and continue on its own branch. Keep this audit unopened.

## Blockers

None identified. CPU-only is adequate for the small character model; throughput and wall time remain measured outcomes.

## Decisions / rulings

- Draw19 selected MA-464 uniformly from a frozen 542-row eligible pool at index 78. Replay source is `source/random_draw.json`.
- PA83 establishes AdaMix's stochastic adapter mixture and merged inference as the closest native baseline. This screen implements a small nanoGPT-scale counterpart, not the source paper's large-model evaluation.
- Four preflight tests passed, including adapter gradient and exact fold/merge checks.
- Corpus prefixes were used for the frozen development screen; audit bytes were not decoded or evaluated.
- Development ran both frozen seeds and all five conditions. The Mirror adapter-only payload is 21,202 B versus AdaMix 36,232 B (58.52%, failing the maximum 55% ratio); total bank is 897,778 B versus 912,610 B (pass). Mirror also misses the 0.02 nat FiLM superiority gate in both seeds; audit remains unopened.
