# MA-494 status

- Status: PROMISING, scoped synthetic noisy-address robustness
- Branch: `research/ma-494-error-correcting-expert-ids-20261009`
- Base: `f5e5311d`; A1 includes registered 3x repetition control; verification commit `ae75b488`
- Fresh: 49420-49422 × seeds 0-2

H: Redundant codes reduce route errors without excessive address storage.

T: 32 experts; binary IDs, ECOC lengths 7/9/11, 3x repetition length 15; nearest-Hamming decoding; p=.0-.2 bit noise.

D: PROMISING. At p=.1, binary accuracy .591; ECOC-11 .816 and repetition .868. Payloads: 2,149B / 2,277B vs 1,957B binary; noiseless accuracy 1.0. Repetition beats ECOC, so no Mirror-specific superiority established.

C: Independent synthetic bit-flip channel is not a trained router or task metric.

U: Real router confusion and task-level consequences.
