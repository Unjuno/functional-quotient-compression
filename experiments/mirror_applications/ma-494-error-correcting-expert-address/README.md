# MA-494 — Error-correcting Mirror expert addresses

Status: **FAIL** for Mirror-specific attribution (development screen; fresh sealed)  
Branch: `research/ma-494-error-correcting-expert-ids-20261008`  
Base commit: `9e5eb1de16b0e4e5982fd28beb666a27f18d489d`  
Prior art: PA95 (Error-Correcting Output Codes)

## H — Falsifiable hypothesis

For a fixed 16-expert bank, a 7-bit Hamming(7,4) codebook will reduce p=.05 corruption route errors to <=0.06 with <=1.01x raw-ID actual inference payload, while preserving all expert modes.

## T — What ran

Two frozen seeds (49401, 49402), each with 10,000 uniformly sampled expert requests per bit-flip rate p=0/.01/.05/.1/.2. Compared raw 4-bit IDs, random 7-bit codebook, native Hamming(7,4), and 3x repetition (12 bits). Every payload includes the full 16-expert matrix bank and packed codebook bits. Route error, output RMSE, decoded mode coverage, actual bytes, decode operations and wall time were measured. Full serialization replay and tests passed. Fresh seeds 49411–49413 remained sealed.

## D — Decision

**FAIL** for Mirror-specific attribution. The frozen robustness/byte feasibility gate passed: Hamming route error was 0.0464/0.0448 at p=.05 and payload was 17,322 B versus raw ID 17,307 B. However, this is exactly the standard native ECOC method (PA95), so no Mirror-specific gain is established. Fresh stayed sealed.

## Facts

- At p=.05: raw-ID route error 0.1885/0.1911; random7 0.1352/0.1358; Hamming(7,4) 0.0464/0.0448; repeat3 0.0291/0.0294. Every method retained 16/16 decoded-expert coverage.
- Hamming output RMSE was 0.3027/0.2995 at p=.05, versus 0.6110/0.6267 for raw IDs and 0.2369/0.2435 for repeat3.
- Payloads: raw ID 17,307 B; random7 17,320 B; Hamming 17,322 B; repeat3 17,328 B. All include the identical charged expert bank.
- At p=.01, Hamming route error was 0.0018/0.0031; at p=.1 it was 0.1492/0.1546. Repetition is more robust at these rates but spends 12 bits.
- No-noise upper had zero route error/output RMSE. Metric replay maximum difference was zero.

## Interpretations

A small redundant address can sharply reduce route errors under bit corruption for only 15 extra serialized bytes in this fixed bank because the shared expert matrices dominate payload. Hamming(7,4) offers a useful robustness/bit frontier versus random 7-bit IDs and raw IDs, while 3x repetition gives lower error at p=.05 for five more bits. Both are standard error-correcting codes, not Mirror-specific capacity gains.

## C — Strongest counter-hypothesis

The corruption channel and uniformly selected fixed expert bank are synthetic; a learned router may produce structured errors rather than independent bit flips. Matrix output RMSE also weights expert-pair differences, so it is not a natural task quality measure.

## U — Still unknown

Robustness under learned router confusion, domain shift, burst errors, neural expert outputs and deployment-specific routing latency remains untested. The study does not increase expert count or independent model capacity.
