# MA-1179 STATUS

- Stage: **FAIL** — primary preregistered gates both failed, but mechanism and replay integrity PASS.
- Date: 2026-10-25.
- Branch: `research/ma-1179-correlated-teacher-core-warmstart-20261025` (do not merge automatically).
- Teacher training: 35/35 ordinary pretrained MLPs complete; positive control derives 3 teacher views per seed.
- Students: 120/120 complete, 750 total updates each; all SHA256 sealed before Fresh/OOD.
- Paired primary correlated-world gates: Mirror warm >10% vs cold **1/5 FAIL**, vs BOTH warmed FiLM and rank1 **0/5 FAIL**.
- Integrity: 7/7 source tests PASS; 120/120 student checkpoint replay PASS with exact Fresh score reproduction; 5 teacher bundle hashes match.
- Frozen protocol SHA256: `8ef02a94680de606cef70655a0b809701c2371f8eb393af6c0928d41581fd16b`.
- Runner SHA256: `472f330c44fe491daa30bb80308c1962a7215641c2b15a534fae7b2ed34f87b2`.
- Results CSV SHA256: `4aec9bfc54443372126a87b90daa22d0933f8de13cea8390acd0eb9c69af0480`.
- CPU FP32 single torch thread (EPYC 9V74, PyTorch 2.10.0+cpu).
- Evidence and full limitations in `REPORT_JA.md`.
- No changes to MA-1178 or `main`.
