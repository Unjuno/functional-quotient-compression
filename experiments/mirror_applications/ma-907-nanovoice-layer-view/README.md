# MA-907 — factorized voice × layer NanoVoice View

Status: **UNTESTED — native baseline unavailable**
Base commit: `68465c691e82ff1f89e94ba763a0e26dfa9642cf`

## H

Factorizing speaker identity × decoder layer modulation may reproduce useful per-layer speaker adaptation with fewer serialized bytes than full per-speaker scale tables, while improving on native shared-scale and simple factor controls.

## T / D

PA253 and official-source availability were reviewed. Exact-title bounded GitHub search found no NanoVoice implementation; no source, checkpoint, or speaker corpus is available locally. Required held-out speaker×layer quality/bytes cannot be evaluated without those assets. No training or scoring was done; UNTESTED, not FAIL.

## C / U

Native scale matrix and ordinary speaker embeddings may already provide identical factors. Official source/checkpoints and held-out speech quality remain unresolved. See `EXECUTION_BLOCKER.md`.
