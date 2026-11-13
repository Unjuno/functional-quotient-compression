# MA-984 — RIS phase control codes

Status: FAIL for Mirror-specific attribution. Frozen synthetic RIS control test; matched native phase-gradient control aliases Mirror exactly.

## H — hypothesis

Two task coordinates controlling a low-rank phase profile can retain most continuous RIS beamforming quality with fewer serialized phase-command bits than native per-element control.

## Prior-art delta

PA290 already uses programmable RIS phases and model-driven joint beamforming. This screen tests only an ideal 64-element narrowband reflected path; full-element 2-bit control and native group/phase-gradient controls are required.

## T — protocol

For 128 independently generated single-user links per fresh world, derive the continuous phase-alignment target, then compare full continuous, full-element 2-bit, eight-group 2-bit, two-coordinate quantized Mirror, and the same native two-coordinate phase gradient. Every code and decoder header is bit-packed into the measured command payload. Controller/actuator counts are proxies.

## Limits

No 3GPP/PA290 reproduction, multi-user interference, joint AP optimization, measured RIS, energy, switching latency, or fabrication result.
