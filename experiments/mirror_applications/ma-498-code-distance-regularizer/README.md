# MA-498 — Learned code distance regularizer for Mirror bank

Status: **FAIL**.

**H:** At fixed 8-bit address size, development-selected max-min distance reduces fresh noisy misroutes over a random unique code assignment.

**T:** 32 IDs; random unique and development-selected 8-bit codebooks, plus ordinary 5-bit IDs. Fresh A1 worlds 49820-49822 × seeds 0-2; bit flips p=0,.02,.05,.1,.2. A0 duplicate-code random control is preserved and excluded.

**D:** FAIL. At p=.1, selected and random 8-bit codebooks both averaged .667 accuracy, min distance 1, and 2,085B payload. Binary IDs averaged .595 accuracy. Thus redundancy helped over binary IDs, but the development distance selection did not improve over random at equal bits/bytes. At p=.2, both 8-bit codebooks averaged .388 vs binary .330.

**C:** The candidate pool and 8-bit budget produced minimum distance 1; max-min selection had no useful freedom over random. Any gain came from adding three address bits, not the regularizer.

**U:** Structured BCH/ECOC constructions, larger code length and task-level router loss remain untested.
