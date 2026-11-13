# MA-916 execution blocker

## Fact

- PA253–PA256 identify NanoVoice, HyperTTS, Mixture of Adapters TTS and Hyper-MoA as the candidate's strongest direct/shared-speaker methods.
- Bounded exact-title GitHub repository searches found no runnable NanoVoice, MoA-TTS or Hyper-MoA official repositories. HF `nanobee/nanovoice` has no model artifact files.
- HyperTTS official source and an LTS100 checkpoint were located during MA-908, but raw held-out reference audio and speaker embedding arrays were absent; the old dependency stack was not runnable in the current shared runtime.
- LibriTTS is published on HF, but the available test-clean set is four large parquet shards (~1.47 GB). VCTK/LibriTTS availability does not replace the missing matched baseline weights and evaluation pipeline.
- Runtime is CPU-only.

## Decision

Keep UNTESTED. A one-model proxy or handmade adapter bank would not compare the registered set of strong controls or establish deployment quality. Do not substitute unrelated TTS systems or call missing evidence a FAIL.

## Resume condition

Obtain runnable official baseline checkpoints and matched consent-safe evaluation speakers, then freeze the whole-speaker split and compare actual serialized bytes, speaker similarity, intelligibility, adaptation compute and inference latency.
