# MA-882 status

- Status: FAIL
- Branch: `research/ma-882-layer-pair-alignment-20261112`
- Protocol freeze: `ca28acaf8`
- Development ranks frozen: `4350280f0` (PCA 256, CP 8; neither development quality rule met)
- Fresh seeds: 983, 997, 1009 across two ordered pairs
- Quality gates: 0/6 pass; storage gates: 6/6 pass
- Payload reconstruction: full/PCA/CP byte/hash checks pass; max replayed NLL difference 0.00072 nat
- Verification: see `VERIFICATION.json` and `source/verify_results.py`

See [RESULTS.md](RESULTS.md). Top-2 layer matching beats fixed nearest-depth routing in this screen, but compressed map views fail quality.
