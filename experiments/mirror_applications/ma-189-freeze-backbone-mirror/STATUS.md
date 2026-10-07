# MA-189 status

- Status: **PROMISING** for the preregistered aligned-family quality/byte/retention gate
- Branch: `research/ma-189-freeze-backbone-mirror-20261007`
- Protocol freeze commit: `08b8c7b89d38edafe48d1e25b5aeba93a485c7c6`
- Result commit: `ba9639983b09eb75237a6732d1f3c75ad8d75b8a`
- Development complete: yes
- Fresh/audit opened: yes, after selecting LR 0.01 using development rows
- Results committed: yes
- Verification committed: yes (in the tracker commit)
- Registry row updated: yes

## H — hypothesis

On a frozen shared linear base, two Givens coordinates will recover a task in the corresponding input/output view family within 1.10x rank-2 LoRA quality and at <=0.25x its incremental inference bytes, while unrelated maps require private parameters.

## T — execution

One new 16D->8D skill after an exact frozen Task-0 synthetic base. Aligned teacher applies input and output first-pair rotations; unrelated teacher is an independent map. Budgets 16/64/256 examples, 200 full-batch updates. Development seeds 18901/18902 selected LR .01; fresh 18911–18913. Controls: hard tie, one-sided view, rank-1/2 LoRA, nonzero-initialized rank-2 hypernetwork, independent full weights.

## D — decision

**PROMISING** on this constructed aligned family. At 64 examples, quality/retention/bytes passed 3/3 fresh worlds. Mean aligned MSE: two-sided Mirror 8.87e-11 vs rank-2 LoRA 6.75e-3. Incremental inference payload 55 B vs 228 B (0.241x); full payload 684 B vs 851 B. Task-0 MSE did not change. Active-compute proxy was 0.773x LoRA, but eager training wall was ~2.25x and inference throughput ~0.20x LoRA. Unrelated tasks remained poor and favored private weights.

## C — strongest counter-hypothesis

The aligned teacher exactly matches the two-angle candidate, while rank-2 LoRA does not represent the combined view as directly and only 200 updates were allowed. Rank-4 LoRA was not tested.

## U — unresolved

No rank-4 LoRA or byte-matched generic shared-basis control; no optimized runtime; no natural tasks or capacity frontier. Mirror-specific superiority beyond this exact teacher family is not established.
