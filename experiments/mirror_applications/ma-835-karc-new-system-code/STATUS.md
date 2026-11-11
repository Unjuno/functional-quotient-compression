# MA-835 status

Status: **FAIL** — scoped synthetic mechanism/storage screen. Fresh data remained sealed after the rank-2 quality gate miss and exact native factorization alias.

## H / T / D / C / U

- **H:** A compact system-specific Mirror coordinate over frozen KARC features would attain rollout nRMSE <=0.02, save >=25% complete bytes against independent readouts, and beat native readout adaptation.
- **T:** One development seed (83501); eight 4D synthetic systems; 48 support transitions and 32 heldout rollouts per system; horizon 24; no optimizer updates. Compared shared-only, independent readouts, native rank-2 factorization, Mirror rank-2 codes, and cubic basis growth. Complete compressed NPZ bytes charged, including feature IDs/specification and readouts/codes. Focused test: 1 passed.
- **D:** FAIL. Independent readouts reached nRMSE 0.000228 at 2,690 B. Rank-2 Mirror/native factorization both used 1,469 B but reached 0.038621, above the 0.02 gate; payload SHA-256s and active operations match exactly. Adding cubic basis features reached 8.45e-9 but cost 3,206 B, above independent storage. Shared-only was 902 B with nRMSE 0.28187. All controls used 104,448 active feature evaluations; no optimizer updates. Fresh seeds were not opened.
- **C:** The apparent compression is rank-2 readout factorization, an ordinary native representation. The quality miss comes from out-of-bank cubic dynamics; adding private/expanded features restores quality but erases the storage advantage.
- **U:** No trained KARC, real continual-learning, real deployment, or unseen-system-code generalization evidence. Support-derived system readouts are stored at inference. Wall-clock is a single local synthetic inference timing and is not a stable performance claim.

## Fact / Interpretation / Hypothesis

- **Fact:** The Mirror and native rank-2 payloads have identical bytes (1,469), SHA-256, error, and active feature count. The independent readout meets the quality threshold at 2,690 B; basis growth reaches near-zero error at 3,206 B.
- **Interpretation:** The Mirror label adds no function or storage benefit over native factorization. The frozen basis has a real coverage limit in this constructed world; private basis growth is required for exact recovery and costs more than the independent readout.
- **Hypothesis:** On new-system dynamics, stored readout codes only help when the frozen feature span already covers the new system. Under this test, the code-only approach does not preserve quality for the shifted systems.
