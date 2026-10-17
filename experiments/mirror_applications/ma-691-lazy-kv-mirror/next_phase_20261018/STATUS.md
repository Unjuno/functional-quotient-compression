# MA-691 cache-aware end-to-end experiment plan — 2026-10-18

- **Plan recorded only.** This folder contains no new training, end-to-end decode, GPU benchmark, or new fresh result.
- Existing MA-691: PROMISING exact cache algebra; retained.
- Existing MA-253: FAIL to replace independent LoRA experts; cache-placement PASS; retained.
- Existing MA-1164: CPU fused implicit Mirror implementation is a separate mechanism/runtime result; no natural-language speed/quality win.
- Local MA-691 follow-up diagnostics: CPU FP32, one thread, 4 heads x 32 dimensions, prefix 8,192, 8 logical Views; canonical cache 8 MiB vs eight materialized caches 64 MiB; eager CPU single-token read varied across experiments (~2.5 ms canonical versus ~4.0 ms materialized in one run). The native hard-shared K/V Query-only control has similar or lower latency but does not reproduce full View functionality; this is not a trained model comparison.
- L0 protocol: \`PROTOCOL_L0_FROZEN.json\`, fixed before newly numbered fresh seeds are opened.
- L1 trained natural-language protocol: **DRAFT, NOT FROZEN**, requires dataset/tokenizer hashes and implementation preflight.
- No edits to \`main\`, the MA registry, or historical result statuses.
