# MA-1033 amendments

## A1 — Correct full frozen-CLIP forward before fresh audit (2026-11-15)

The first development smoke run applied prompts but forwarded only the first transformer layer in each CLIP branch. The result was clearly invalid for the intended frozen-CLIP baseline: zero-shot reached 80.6% overall accuracy while all learned-prompt arms collapsed to 10.0% overall (the independent-cell arm reached 10.3%). This code path did not implement the registered backbone and is excluded from the experiment's scientific comparison.

Before any fresh audit seed was accessed, both text and vision prompt paths were corrected to run every frozen CLIP transformer layer. Unit tests passed and the development seed 103300 was rerun from scratch with the protocol's fixed 20 updates. The corrected source hash and development result hash are recorded in `FREEZE.json`. No audit outcome was observed before this correction. The first invalid smoke measurements are retained here as a code-path diagnostic rather than silently discarded.
