# MA-322 status

- Status: FAIL — strict storage promotion gate missed; quality passed.
- Branch: `research/ma-322-tt-core-mirror-adapter-bank-20261008`
- Base commit: `a9991e1`
- Fresh seeds 32211–32213 complete: aligned Mirror 5,048B vs direct 5,074B (0.51% lower); mixed Mirror 25,464B vs direct 25,426B (0.15% higher), with 32 private fallbacks in both. All max test nMSE <=6.44e-7. Mirror fit proxy 1,610,612,736 vs 1,048,576 (~1,536x). FAIL under frozen >=10% byte gate.
- 30 payloads reloaded and byte/hash checked; 30 summary rows and 1,536 allocations replayed exactly; 5 tests pass. See README H/T/D/C/U report.
