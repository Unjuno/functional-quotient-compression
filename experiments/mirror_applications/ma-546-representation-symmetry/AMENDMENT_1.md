# MA-546 protocol amendment 1 — inverse orientation and floating-point output gate

Date: 2026-10-09 UTC

## Reason

The first development run revealed that the dense orthogonal control applied Q rather than Q^-1 to the down-projection. It also showed that a strict absolute max-logit threshold can fail from float32 accumulation order even when top-1 is unchanged and KL is tiny (monomial: max logit delta 0.0088, top-1 agreement 100%, max KL 2.6e-6). No fresh world was accessed.

## Change

Correct dense compensation to W_down @ Q^T for z @ Q^T. Retain local projection max-delta <=1e-5. Gate full-model equivalence by top-1 agreement 100%, mean KL <=1e-5 and max KL <=1e-4; continue reporting raw max-logit difference without using it as the sole criterion. The definition remains an exact inverse-coordinate gauge. All task prompts, seeds, transforms, model, layer and storage rules remain fixed.

## Data handling

Preserve the first development output under `results/pre_amendment_1/`. Rerun both development seeds under corrected code. Fresh seeds remain unopened.
