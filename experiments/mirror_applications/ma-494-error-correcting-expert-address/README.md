# MA-494 — Error-correcting Mirror expert addresses

Status: SCREENING  
Branch: `research/ma-494-error-correcting-expert-ids-20261008`  
Prior art: PA95 (Error-Correcting Output Codes)

## H — Hypothesis

Redundant error-correcting expert codewords can reduce wrong logical-expert routing under address-bit corruption, trading a small number of paid address bits for function quality and route reliability.

## T — Frozen protocol

A 16-expert linear function bank is held fixed across raw 4-bit IDs, same-length random 7-bit addresses, Hamming(7,4) ECOC and 3x repeated 4-bit addresses. Inject independent bit flips at p=0/.01/.05/.1/.2. Measure routing error, output RMSE, coverage, codebook+expert-bank NPZ bytes and decode operations. Fresh worlds/seeds remain sealed unless the frozen gate passes.

## D — Pending development runs

Fresh seeds 49411–49413 remain sealed. ECOC is a standard native robust-address control; robustness from redundancy is not a Mirror-specific gain unless it exceeds equal-bit native coding.
