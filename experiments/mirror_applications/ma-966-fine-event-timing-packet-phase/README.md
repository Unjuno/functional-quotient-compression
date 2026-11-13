# MA-966 — Fine event timing under shared Mirror phase views

Status: SCREENING. Protocol frozen before local dataset extraction/scoring.

## Hypothesis

A shared event encoder with one small phase coordinate per temporal window will improve sign decoding on unseen Vernier offsets over the same shared encoder, preserve exact timestamp access, and use substantially fewer serialized bytes than four independent encoders.

## Mirror insertion

The shared event Transformer receives four windows of 48 events from each real D1 Vernier event trial. A learned phase coordinate `m_window` rotates a periodic sin/cos time feature pair for the corresponding window. Raw timestamp remains an input channel. The cheapest control applies a native time offset before the exact same periodic encoding; this is an expected algebraic alias and the protocol treats that as a Mirror-specific FAIL if matched.

## Prior-art delta

PA285 demonstrates precise event timing on tiny images and Vernier-style subpixel discrimination. Its public data includes 708 real D1 Vernier trials and 14 signed offsets. Official loader has periodic time encoding and affine offset/scaling. This screen asks whether a compact shared-window phase code adds anything beyond those direct controls. The published model used MNIST-pretrained PointTransformer features; this experiment uses a small from-scratch CPU Transformer and is not a reproduction of that published result.

## Results

See `RESULTS.md` after the locked fresh evaluation. All phase/time codes and model metadata are charged in serialized payload bytes.
