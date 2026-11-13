# MA-911 — causal speaker-content subspace Mirror factors

Status: **UNTESTED — required interventional speech assets unavailable**
Evidence lane: SPEECH_REPRESENTATION / CAUSAL_INTERVENTION / OOD_SPEAKER / STORAGE
Base commit: `9e779e4bbf510ad24b9b88f17a887c38e7cc8d9f`

## H

Compact speaker/content Mirror axes should preserve independent identity and lexical content under heldout swaps and OOD speaker tests, beyond the official native interventional subspace.

## T / D

Official `mjukus/interventional-post-training-speech` source was found. Its required LibriTTS-derived F5-TTS synthetic intervention data, WavLM features, SpeechCommands and VoxCeleb1 audio are absent; the generation recipe requires an Apptainer F5-TTS image. No model or proxy was run. Remain UNTESTED, not FAIL.

See `EXECUTION_BLOCKER.md` and `source/UPSTREAM.md`.
