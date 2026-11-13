# MA-942 — DCVC-UF frame × QP factor

## H — hypothesis

A shared factor over frame slot and QP can replace DCVC-UF’s native 64-QP module bank while preserving equal-coded-rate quality and parallel decode FPS.

## T — work completed

Reviewed PA269/PA270 and the DCVC-UF paper. MA-941 confirmed the official paper is reachable but no public code/checkpoint/entropy coder/bitstream was found. The current environment has no CUDA device. MA-942 further requires the native 64-QP rate bank and actual equal-rate coded bits, neither of which is available. No proxy run was conducted.

## D — UNTESTED

The required rate-distortion and decode measurements cannot be computed without the native codec and its QP bank.

## C — strongest counter-hypothesis

A synthetic QP-conditioned decoder could test generic factorization, but cannot establish codec BD-rate or native rate-control equivalence.

## U — unconfirmed

All QP-state bytes, equal-rate quality, bitrate, FPS and frame-slot interaction results.
