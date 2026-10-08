# MA-282 status

- Status: PROMISING (narrow aligned synthetic screen)
- Branch: `research/ma-282-monarch-mirror-ffn-20261008`
- Base commit: `28194ea`
- Last verified commit: pending commit
- Development complete: yes; LoRA rank 4 selected from ranks 1/2/4 on development only
- Fresh/audit opened: yes; final seeds 28241–28243
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Result

- Aligned Mirror: MSE 0.0 in 3/3 fresh worlds; 925 B vs independent full 3,279 B (0.282×).
- Free-angle Monarch: same exact quality at 1,145 B; hard tie 841 B, MSE 3.09; rank-4 LoRA 2,373 B, MSE 1.13.
- Unrelated task maps: Mirror mean MSE 2.35; independent full MSE 0 in 3/3.
- Three invalid preliminary fresh runs are retained but excluded. Final seeds and amendment details are in PROTOCOL.json.

## Verification

`python -m unittest discover -s experiments/mirror_applications/ma-282-monarch-mirror-ffn/tests -v`: 7 passed. All 42 final fresh metrics replayed with max absolute difference 0.0; serializer roundtrip and exact payload accounting verified.

## Next action

Update registry, claim ledger and status board; commit and push this research branch.

## Blockers

None for this fixed CPU mechanism/storage/runtime screen. No GPU/Transformer claim.
