# MA-416 — DeepSDF latent codes versus structured Mirror geometry codes

Status: SCREENING; protocol frozen before training. Prior art PA68 (DeepSDF shared decoder and per-shape latent codes).

## H — Hypothesis
For instances that are translated, rotated and scaled versions of a canonical shape, a five-coordinate geometric View may reconstruct new fields and interpolate between instances with fewer serialized bytes than unconstrained DeepSDF latent codes.

> **Mirror insertion:** this experiment adds a five-value per-instance coordinate `m=(tx,ty,log_rx,log_ry,theta)` before a shared neural implicit decoder, so many transformed shape fields reuse one canonical decoder.

## T — Protocol
The task is 2D ellipse signed-distance reconstruction. The Mirror decoder is pretrained on a canonical circle and then frozen; DeepSDF jointly trains a shared conditional decoder and unconstrained five-dimensional instance codes. Both adapt held-out codes using only 256 support points per shape and evaluate on disjoint query points. An explicit native affine-coordinate control uses the same geometry transform to audit whether any result is Mirror-specific. An analytic ellipse function is the quality oracle. Interpolation uses midpoint instance codes and compares against the corresponding midpoint geometry. Exact seeds, updates, payload format, metrics and gates are frozen in `PROTOCOL.json`.

Fresh seeds 41611–41613 remain sealed until both development seeds pass. Actual compressed NPZ bytes include the decoder and every per-shape code.
