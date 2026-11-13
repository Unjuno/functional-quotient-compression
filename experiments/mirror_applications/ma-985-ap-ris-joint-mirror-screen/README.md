# MA-985 — AP×RIS joint factorization screen

Status: FAIL for the rank-2 synthetic quality/storage gates; native product control aliases Mirror.

## H — hypothesis

A shared AP beam basis crossed with a low-dimensional RIS phase basis can encode held-out joint wireless configurations with less serialized control state while retaining weighted sum rate.

## T — intended controls

The CPU simulator compares standard full AP WMMSE plus full RIS phase-gradient updates, AP-rank-2 and RIS-rank-2 ablations, the combined Mirror factorization, and an ordinary native AP×RIS product-code control. Each channel configuration is optimized directly; this is not a learned policy tested on held-out identities. The WMMSE AP subproblem enforces a total transmit-power limit; RIS phases are updated by gradient ascent at fixed AP precoders, retaining the best observed iterate.

## Prior-art boundary

PA290 already studies joint MU-MISO WMMSE and model-driven RIS beamforming. This is a small synthetic algorithm screen only; it does not reproduce PA290's trained graph network or 3GPP scenarios.

## Limits

No measured channels, 3GPP propagation, real RF hardware, energy or deployment latency claim.
