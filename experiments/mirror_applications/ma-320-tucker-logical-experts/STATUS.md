# MA-320 status

- Status: **FAIL** on the frozen 10% storage promotion gate.
- Branch: `research/ma-320-tucker-logical-experts-20261008`
- Protocol frozen: `3f23869`; pre-fresh source/results frozen: `7612211`.
- Fresh seeds: 32011, 32012, 32013.
- Mirror used 35,886B versus 37,074B for rank-16 Tucker with the same private fallback (3.2% fewer, below 10%) in all three seeds. All expert test nMSE stayed below 1e-4. 16/64 unrelated experts used full private matrices.
- Mirror fit proxy was 1.61B vs 50.3M for Tucker (~32x). Replay: 45 summary rows and 2,240 allocation events exact; five tests pass.
- Next P0: MA-322.
