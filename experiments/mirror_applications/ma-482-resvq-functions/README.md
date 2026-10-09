# MA-482 — Residual-VQ Mirror logical-function addresses

Status: **FAIL** (development screen; fresh data sealed)  
Branch: `research/ma-482-resvq-mirror-address-20261008`  
Base commit: `1b6ccdf9d5707529507d6bc5ee49e97133337b49`  
Prior art: PA92 (SoundStream / EnCodec residual VQ)

## H — Falsifiable hypothesis

For the frozen rank-four function bank, two to four train-fit residual-VQ stages will reach heldout RMSE <=0.05, >=90% distinct heldout address tuples and <=6,101 actual payload bytes in both seeds, improving over single-stage VQ while retaining int8-scale storage.

## T — What ran

Two frozen development worlds (48201, 48202): 192 linear functions over 16-dimensional inputs and outputs; four shared matrix atoms fit on 128 functions; 64 heldout functions evaluated on 64 fixed queries each. Compared independent matrices, continuous codes, int8, single VQ K=16/64/128, residual VQ K=8/16 at depths 2/3/4, and a native residual-VQ control. Each residual codebook fit only training residuals. Full uncompressed NPZ bytes, stage-index tuple diversity, heldout output RMSE, operation proxies and fit/decode/query wall times were recorded. Independent upper reproduces targets; verifier replay and tests pass. Fresh 48211–48213 remain sealed.

## D — Decision

**FAIL**. Residual stages improved the quality-rate curve over one-stage VQ, but no residual setting met both the frozen <=6,101-byte int8 payload and <=0.05 RMSE in both seeds. RVQ8x4 narrowly missed quality in seed 48202 and used 11,934 B; RVQ16x4 passed quality in both seeds but used 12,450 B. Native RVQ is byte/output identical to Mirror RVQ, so the measured improvement belongs to ordinary residual VQ, not Mirror-specific coordinates.

## Facts

- Best near-gate compact residual: RVQ8x4 used 11,934 B, distinct-address fraction 0.969 in both seeds and RMSE 0.04400 / 0.05002.
- RVQ16x4 used 12,450 B, distinct-address fraction 1.0 and RMSE 0.03118 / 0.04038.
- Int8 used 6,030 B, distinct-address fraction 1.0 and RMSE 0.00281 / 0.00129. Continuous codes used 8,088 B with approximately zero error; independent upper used 197,279 B with zero error.
- Single VQ128 used 8,832 B with RMSE 0.06852 / 0.08420. Residual VQ improved this single-stage VQ frontier at higher storage and fit work.
- Every Mirror/native RVQ pair had identical NPZ bytes and SHA256; replay maximum difference was zero. Fresh remained sealed.

## Interpretations

Residual codebooks recover quality and address uniqueness more effectively than one-stage VQ on this synthetic function bank, but their extra centers and indices push payload above int8 while error remains higher. The improvement is completely explained by native residual VQ.

## C — Strongest counter-hypothesis

The int8 control is especially strong on this smooth rank-four coefficient family. More residual stages or different per-stage code sizes could recover near-lossless quality, but added centers/indices and fitting operations may erase the storage advantage. A task with multimodal or non-smooth function populations could shift that frontier.

## U — Still unknown

Whether residual codes help expert functions under learned routing, or on natural neural tasks, remains unknown. MA-481 and MA-482 repeat the standard-codebook attribution cause; see the family diagnostic. MA-484 is only eligible as a redesigned expert-task experiment with routed behavior and native MoE/RVQ controls.
