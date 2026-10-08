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

## Results and decision

**D — FAIL.** Fresh mean packet NRMSE over three worlds × three seeds: shared recurrence 0.134059 / 1,860B; Mirror packet View 0.000225 / 2,125B; ordinary latent 0.006033 / 2,518B; independent offset transitions 0.000068 / 2,485B. Mirror is much better than the shared transition and latent control, but is 3.3× worse than independent and uses 85.5% of its bytes, missing both preregistered gates. Spectral radii remained below 0.811. Mirror’s measured CPU rollout time was 0.000234s vs 0.000150s shared recurrence; this implementation does not establish parallel decode speedup.

**FACT:** the packet View improves sharply over shared recurrence, but does not recover independent quality or the required byte reduction. The ordinary latent control is weaker and larger. Same four recurrent transitions were used per method.

**INTERPRETATION:** a low-dimensional packet plan can add useful function over a tied transition in this aligned task, but it does not replace independent offset functions at the registered cost/quality frontier. Extra logical outputs do not count as independent capacity.

**H:** tested whether a per-packet View makes multiple future offsets from one shared SSM useful at lower bytes.

**T:** 4D stable transition, packet width 4; 400 AdamW updates × batch 96; development worlds 43900/43901 selected LR 0.003 for shared/latent/independent and 0.01 for Mirror; fresh worlds 43910/43911/43912, three seeds; actual payload, packet NRMSE, MAC proxy, wall time, and stability.

**C:** teacher dynamics are aligned to Givens rotations, and the result uses oracle role labels; natural packet prediction could behave differently. The implementation rolls the packet sequentially, so it is not evidence for actual parallel-token latency.

**U:** natural language, valid-path retention, learned plans/routing, parallel kernels, and accelerator throughput remain untested.
