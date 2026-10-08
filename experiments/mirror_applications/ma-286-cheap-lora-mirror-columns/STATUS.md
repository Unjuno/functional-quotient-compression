# MA-286 status

- Status: FAIL for the registered Mirror-specific margin; shared-basis screen passes
- Branch: `research/ma-286-cheap-lora-reconciled-20261008`
- Base commit: `28194ea`
- Development complete: yes
- Fresh/audit opened: yes; 28611–28613
- Results committed: yes (`ed10296`)
- Verification committed: yes
- Registry row updated: yes on cumulative branch

## Result

- Mirror index: 4412 B, mean normalized MSE 7.310e-16; eight aligned addresses, two private low-rank fallbacks.
- Matched task-local Cheap-LoRA: 6580 B; independent full: 41279 B.
- One-hot shared-factor control: 4490 B, same functions and quality; index saves 78 B (1.74%), below the preregistered 10% Mirror-specific gate.
- Unrelated functions use private rank-4 factors.

## Verification

`python -m unittest discover -s experiments/mirror_applications/ma-286-cheap-lora-mirror-columns/tests -v`: 5 passed. All 24 fresh metrics replay exactly; payload SHA256 and serializer roundtrip checked.

## Next action

No further action; status and evidence are indexed on the cumulative branch.

## Blockers

None for this synthetic fixed-address CPU screen.
