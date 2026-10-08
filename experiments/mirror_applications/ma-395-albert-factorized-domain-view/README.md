# MA-395 — ALBERT factorized embedding + domain Views

Status: frozen development protocol. Base commit: `bb638b6327323c265a1a0d602a9d7964a073ab06`. Prior art: PA61.

## H

Test whether a domain Givens coordinate inside a shared 4D token embedding bottleneck improves held-out token-domain function quality beyond ALBERT-style hard sharing and domain adapters applied after projection.

## Controls and scope

Controls: independent domain-token embeddings, shared factorized embedding, post-projection FiLM, rank-2 post-projection LoRA, and a bottleneck Mirror view. All private state and inference bytes count. Two development seeds use a locked 25% held-out pair split; fresh seeds remain sealed unless every registered gate passes.

This aligned synthetic screen does not establish natural-language, multilingual, or independent-capacity claims. See `PROTOCOL.json` for exact conditions and gates.
