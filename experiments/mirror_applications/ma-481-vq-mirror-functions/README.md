# MA-481 — VQ Mirror address codebook for logical functions

Status: **FAIL** (development screen; fresh data sealed)  
Branch: `research/ma-481-vq-mirror-logical-functions-20261008`  
Base commit: `58ed65ebb5395d36b3b5cb55929ea2eca4cad232`  
Prior art: PA91 (VQ-VAE)

## H — Falsifiable hypothesis

On a rank-four bank of 192 linear functions, train-only VQ address codebooks at K=16/64/128 will keep heldout function RMSE at or below 0.05, retain at least 90% distinct heldout addresses, and use at most 25% of independent full-matrix payload bytes. A failure of any required clause rejects this frozen screen.

## T — What ran

Two frozen development worlds (48101, 48102), each with 128 training and 64 heldout functions, 16x16 matrices, four shared atoms and 64 fixed query inputs per function. Compared independent matrices, continuous shared-basis codes, int8 codes, VQ16/64/128 and corresponding exact native VQ implementations. VQ fit used training coefficients only. Full actual uncompressed NPZ payloads, fit/decode/query wall times and operation proxies were recorded. The independent upper reproduces all targets; verifier replay and serialization hashes pass. Fresh seeds 48111–48113 were not opened.

## D — Decision

**FAIL**. No VQ size met heldout RMSE <=0.05 or >=0.90 heldout address uniqueness in either seed. VQ payloads were smaller than int8, but RMSE was substantially worse. Native VQ was byte- and output-identical to the proposed Mirror address, so no Mirror-specific attribution is available.

## Facts

- VQ16: 5,771 B; RMSE 0.1044–0.1179; unique heldout addresses 0.219–0.234.
- VQ64: 6,539 B; RMSE 0.0744–0.0859; uniqueness 0.594–0.672.
- VQ128: 7,564 B; RMSE 0.0668–0.0747; uniqueness 0.719–0.750.
- Int8: 6,101 B; RMSE 0.00115–0.00157; uniqueness 1.0. Continuous codes: 8,149 B and effectively zero RMSE. Independent full matrices: 197,333 B and zero RMSE.
- For all K and both worlds, native VQ payload bytes and SHA256 exactly match Mirror VQ. Metric replay maximum difference is zero.
- The accounting-only `AMENDMENT_1.json` records the missing decode/index compute fields. The original first-run outputs remain under `runs/pre_amendment_dev_*`; amended development reruns are authoritative.

## Interpretations

The bit saving from this VQ codebook comes with collisions and function error on this rank-four family. In this screen, int8 is a materially better quality/storage point despite costing 330 B more than VQ128 and 330 B more than VQ16 in the respective comparisons. VQ is also an ordinary native codebook representation, not Mirror-specific. These conclusions apply only to the synthetic function-bank protocol.

## C — Strongest counter-hypothesis

A rank-four family may make continuous or int8 coefficient sharing unusually favorable, while this fixed low K range is too coarse for VQ. A larger codebook could improve approximation, but its paid center/index bytes and heldout address collisions would have to beat int8 and meet the frozen quality criteria; this experiment does not establish that.

## U — Still unknown

Residual VQ may shift this quality-rate curve; that is the separate MA-482 hypothesis. Natural neural function banks and trained model behavior remain untested. No capacity claim follows from the nominal codebook size or possible code combinations.
