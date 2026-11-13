# MA-983 — shared CSI decoder with domain Mirror code

Status: SCREENING. Protocol frozen before fresh test-channel scoring.

## H — hypothesis

A low-dimensional site/frequency coordinate on a shared CSI decoder can restore reconstruction and beamforming quality on an unseen channel domain at fixed 130-bit feedback, using less persistent state than ordinary domain adapters.

## Prior-art boundary

PA291's CsiNet already compresses CSI to low-dimensional feedback codes. This screen uses a tiny linear CsiNet-style model and synthetic channel tensors; it is not a CsiNet reproduction. Native domain embedding, grouped FiLM, low-rank decoder residual and an independent decoder are required controls.

## T — protocol

Train one shared 128-real-to-16-real feedback encoder/decoder on four source domains. For each held-out target domain, fit adaptation using 64 calibration channels and score 256 separately generated test channels. Every method uses the same quantized 128-bit latent plus 2-bit domain identifier. The domain code is persistent at the decoder and paid in serialized state.

## Limits

The channel generator is Saleh-Valenzuela-inspired, not a standard-compliant or measured trace. No field generalization, PA291 result, multi-user rate, energy or latency claim is admissible.
