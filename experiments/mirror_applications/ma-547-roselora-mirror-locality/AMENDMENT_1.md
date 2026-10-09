# MA-547 protocol amendment 1 — exact UTF-8 key router

Date: 2026-10-09 UTC

## Reason

Initial development seed 54701 produced route accuracy 0 because isolated key tokenization did not match the same visible key embedded within prompts under BPE boundary rules. No fresh data was accessed. The run could not evaluate the edit hypothesis.

## Change

Replace token-subsequence routing with exact UTF-8 key-string substring matching over each prompt. Serialize and charge the padded UTF-8 key strings and lengths in every routed inference payload. Edit vectors, target/old IDs, prompts, seeds, controls, margin target and outcome metrics are unchanged.

## Data handling

Preserve the invalid initial development output under `results/pre_amendment_1/`. Rerun both development seeds under this amendment. Fresh seeds remain unopened.
