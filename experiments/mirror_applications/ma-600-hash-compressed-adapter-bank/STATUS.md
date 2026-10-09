# MA-600 status

- Status: **FAIL (development)**
- Branch: `research/ma-600-hash-compressed-adapter-bank-20261009`
- Development worlds 60001/60002/60003 complete; fresh 60011/60012/60013 sealed.
- Mirror: 3,314 B and NRMSE .504/.420/.408; independent rank-2 LoRA: 4,318 B and .0027/.0105/.0139; shared salted hash: 2,586 B and .594/.520/.497.
- Verification passed: 21/21 payloads replay byte-exact; 3 tests passed.

## H / T / D / C / U

- **H:** A shared hashed adapter plus small task Givens codes would represent eight functions at lower bytes and near-LoRA quality.
- **T:** Eight independent seeded rank-2 targets on digits inputs; seven adapter baselines; 1,200 updates; three dev worlds; actual serialized payload bytes.
- **D:** **FAIL.** NRMSE quality gate missed in all worlds; hash compression creates useful representations but high interference. Independent rank-2 LoRA is only 1,004 B larger and dramatically more accurate.
- **C:** The chosen targets are independent rank-2 operators, which naturally favor private rank-2 LoRA factors. Native hash addressing and VeRA controls also show that Mirror does not solve this mismatch.
- **U:** No natural downstream adaptation, pretrained-backbone, near-convergence or GPU runtime result.

## Facts / interpretation / hypothesis

- **Fact:** See `RESULTS_CORE.csv` and `runs/dev_*/metrics.json`.
- **Interpretation:** Shared hash adapters buy storage by sacrificing function quality; private low-rank state is required for this target family.
- **Hypothesis:** Task-specific low-rank function directions are the missing private information that a shared table plus input rotation cannot encode.
