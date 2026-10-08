# MA-482 — Residual VQ Mirror logical-function addresses

Status: SCREENING  
Branch: `research/ma-482-resvq-mirror-address-20261008`  
Prior art: PA92 (SoundStream / EnCodec residual VQ)

## H — Hypothesis

Stacking small residual codebooks can improve heldout logical-function behavior over one-stage VQ at comparable complete serialized bytes, potentially recovering RMSE <=0.05 and >=90% distinct heldout addresses at no more than the int8 code payload.

## T — Frozen protocol

Use the same rank-four 192-function linear bank and 128/64 train/heldout split as MA-481. Compare one-stage K=16/64/128 with K=8/16 residual VQ at depths 2/3/4, continuous codes, int8 and independent full matrices. Fit each stage on train residuals only. Report complete uncompressed NPZ bytes, address-tuple uniqueness, heldout function RMSE and fit/decode/query compute. Exact native RVQ serializer is included as the non-Mirror control.

## D — Pending frozen development runs

Fresh seeds 48211–48213 remain sealed unless the frozen gate passes. Native residual-VQ aliasing is an attribution control: a quality-rate improvement that duplicates ordinary RVQ is not a Mirror-specific gain.
