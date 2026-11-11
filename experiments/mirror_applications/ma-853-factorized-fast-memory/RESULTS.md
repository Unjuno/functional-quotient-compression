# MA-853 — FAIL for Mirror-specific attribution

## H / T / D / C / U

- **H:** Separate key-role and value-role codes compose useful heldout pair functions with less state than storing an independent full matrix for every pairing.
- **T:** Synthetic D=32, four bipolar key roles, four value roles, 16 logical pair maps; 8 checkerboard pairings held out, 64 random queries/pair. Development seed 31; fresh seeds 61 and 79. Compared full pair matrices, native role factors, Mirror pair Views, and MAP/Hadamard bound vectors. 16 role examples, no optimizer updates.
- **D:** **FAIL** for Mirror-specific attribution. Fresh heldout nRMSE is ~2.35e-8 and ~2.72e-8 for full matrices; 0 for native/Mirror/Hadamard. Payloads: full 66,122 B, native 1,864 B, Mirror 1,864 B, MAP/Hadamard 3,400 B. Query MACs over heldout queries: full 524,288; native/Mirror 32,768; MAP/Hadamard 33,024. Native factor roles exactly match Mirror in bytes, outputs and compute.
- **C:** Ordinary native role factorization is the exact and cheapest explanation; MAP/Hadamard also reconstructs roles with a larger payload.
- **U:** Role learning from examples, noisy roles, concurrent sequence writes/interference, trained DeltaNet, and real hardware latency.

## Facts

The 8 heldout combinations are generated from provided key and value role bases; this tests composition/storage, not learned generalization. Full operators pay all 16 matrices; role controls pay both 4×32 bases and all 16 pair indices. Payload arrays are reloaded and compared exactly. Fresh seeds were not used for tuning.

## Interpretation

Composing logical pair functions from shared role bases gives a strong synthetic storage/compute frontier relative to independent matrices, while MAP/Hadamard is a relevant but larger control. Mirror adds no value beyond ordinary factorization of key and value roles. This is not evidence for trained neural capacity or learned role abstraction.
