# MA-540 — Ordered sequential execution of per-step function vectors

Status: SCREENING. PA99/PA72 and SRM003 motivate this test: learned atomic mappings may be composed when an explicit executor passes intermediate state. MA-539 showed the need for a function family that generalizes from support; this experiment uses affine rules closed under composition.

## H — hypothesis

Two ordered primitive FVs executed by one tied shared block will compose held-out affine operators, preserve execution order, and reduce bytes against untied step blocks while beating ordinary operator-ID conditioning.

## T — frozen protocol

Four-bit state space; 24 affine bit-permutation/XOR operators; train on atomic state transitions; hash-split ordered operator pairs with reverse orders kept together. Dev seeds 54001/54002; fresh 54011–54013. Compare FV tied executor, native tied operator IDs, untied blocks, endpoint sum, host two-call, and exact structured upper. Actual bytes and both calls are charged. The complete protocol/freeze were committed before implementation or metric access.

## D

Pending.

## C

The native tied operator-ID executor may exactly reproduce the FV executor; the host loop may be sufficient to explain composition.

## U

Synthetic affine rules, two steps and a small MLP only.
