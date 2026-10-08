# MA-439 — packet-plan Views for SSM future blocks

Status: SCREENING  
Evidence lane: MECHANISM / QUALITY / COMPUTE  
Base commit: `e3aa191`

## H — falsifiable hypothesis

A packet-level Mirror plan code that selects low-description dynamics Views of one shared SSM will predict four dependent future values in parallel with quality near an autoregressive shared SSM, while using fewer bytes than independent per-offset transition heads and improving over an ordinary packet latent of equal dimension.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-example packet plan `m` to the shared SSM transition so that four future offset functions can be evaluated in parallel without storing a separate transition per offset.

- Native method: recurrent SSM next-step predictor.
- Mirror: two-coordinate packet plan selects/rotates shared transition factors for each horizon offset.
- Cheapest control: an equal-width packet latent that conditions four linear outputs, plus direct independent offset matrices.

## Prior art delta

PA10 moves sequence randomness into model inputs for joint packet prediction; PA73 uses input-dependent SSM dynamics. MA-439 tests whether a low-description Mirror plan over a shared SSM transition supplies coherent parallel future functions, with autoregressive and ordinary-latent controls. This is a synthetic mechanism screen, not PTP or language-model evidence.

## Protocol

Four-dimensional stable state; packet width 4; known context features; aligned teacher generated from one shared stable transition plus a packet-specific two-coordinate View. Compare autoregressive shared transition, parallel shared transition, parallel Mirror packet plan, equal-width ordinary latent control, and independent offset transitions. Development worlds 43900/43901 choose LR {0.003,0.01}; fresh worlds 43910/43911/43912, seeds 0/1/2. Same 400 updates and batch 96. Measure packet NRMSE, actual serialized bytes, active MAC proxy and parallel wall time.

PASS: every fresh world Mirror packet NRMSE <=1.10x AR quality and <=60% independent offset bytes; parallel decode wall <=1.25x AR. FAIL if any gate misses or ordinary latent matches Mirror quality at lower bytes.

## C — strongest counter-hypothesis

The packet code may merely be an ordinary latent or offset embedding; the AR baseline may retain a quality advantage from sequential feedback, and parallel work may cost more compute.

## U — unresolved

Natural text token dependencies, valid-path rate, decoding throughput on accelerator, and learned packet routing remain untested.
