# MA-282 status

- Status: PROMISING (narrow aligned synthetic screen)
- Branch: `research/ma-282-monarch-reconciled-20261008`
- Source base commit: `28194ea`
- Source verification commit: `3b9d828`; integrated tests/replay passed
- Development complete: yes; LoRA rank 4 selected from ranks 1/2/4 on development only
- Fresh/audit opened: yes; final seeds 28241–28243
- Results committed: yes on source branch; indexed on cumulative branch
- Verification committed: yes on source branch; rerun on cumulative branch
- Registry row updated: yes on cumulative branch

## Result

- Aligned Mirror: MSE 0.0 in 3/3 fresh worlds; 925 B vs independent full 3,279 B (0.282×).
- Free-angle Monarch: same exact quality at 1,145 B; hard tie 841 B, MSE 3.09; rank-4 LoRA 2,373 B, MSE 1.13.
- Unrelated task maps: Mirror mean MSE 2.35; independent full MSE 0 in 3/3.
- Three invalid preliminary fresh runs are retained but excluded. Final seeds and amendment details are in PROTOCOL.json.

## Verification

`python -m unittest discover -s experiments/mirror_applications/ma-282-monarch-mirror-ffn/tests -v`: 7 passed. All 42 final fresh metrics replayed with max absolute difference 0.0; serializer roundtrip and exact payload accounting verified.

## Next action

No further action; decision indexed on cumulative branch.

## Blockers

None for this fixed CPU mechanism/storage/runtime screen. No GPU/Transformer claim.
