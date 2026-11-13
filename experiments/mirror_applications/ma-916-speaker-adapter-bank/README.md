# MA-916 — speaker adapter-bank storage frontier

Status: UNTESTED
Evidence lane: MULTI_SPEAKER_TTS / STORAGE / QUALITY / RUNTIME
Base commit: `research/mirror-application-worker-ready-20261007`

## H — falsifiable hypothesis

Across matched new speakers, a shared TTS backbone with small Mirror speaker codes can yield a better actual serialized bytes-per-voice versus speaker-quality/runtime frontier than native per-speaker adapters and strong shared methods (NanoVoice scale matrix, HyperTTS generated adapters, MoA and Hyper-MoA).

## T — source and asset review

Reviewed PA253–PA256, exact-title public GitHub and Hugging Face searches, and prior MA-906 through MA-910 feasibility records. No runnable NanoVoice, MoA-TTS or Hyper-MoA source/checkpoint was found. HyperTTS source and one checkpoint exist, but the prior review found missing raw evaluation audio/speaker feature assets and unresolved old dependencies. Public LibriTTS/VCTK data exists remotely but is not present locally; adding a corpus alone does not supply matched runnable controls.

## D — UNTESTED

No model bank, voice adaptation, quality, actual payload bytes, or runtime was measured. This is an execution blocker, not a negative result for the hypothesis.

## C — strongest counter-hypothesis

Native shared speaker methods may already offer the best quality-per-byte, while a code bank either duplicates their conditioning state or degrades speaker fidelity and pronunciation.

## U — unresolved

Runnable official baselines/checkpoints for a matched speaker set, consent-safe fresh voice audio, and a reproducible speaker/content quality pipeline. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
