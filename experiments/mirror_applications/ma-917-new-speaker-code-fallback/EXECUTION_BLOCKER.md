# MA-917 execution blocker

## Fact

- PA253 NanoVoice source/model assets were not located during MA-906/907/916 reviews.
- The official HyperTTS repository has one published LTS100 single-speaker checkpoint. Its VCTK-finetuned checkpoint text is commented out in README.
- VCTK configuration expects a multispeaker model with external 256-dimensional speaker embeddings and GE2E preprocessing. The training configuration defaults to 900,000 steps and batch size 16.
- The repository pins a 2021-era PyTorch 1.7 / TensorFlow 2.5 / librosa stack. Current environment has CPU-only PyTorch 2.14.1.
- No local consent-safe support/query audio or compatible pretrained speaker-embedding weights are available.

## Decision

Keep UNTESTED. The single-speaker LTS100 checkpoint cannot test a new-speaker bank or private fallback, and training a new multispeaker generator on CPU would not be a valid matched evaluation. Public audio alone does not supply the missing generator and speaker encoder.

## Resume condition

Obtain a pinned runnable multispeaker checkpoint and speaker encoder, then freeze support/query speakers and trigger thresholds before comparing Mirror code, private adapter and hybrid fallback in actual bytes, speaker quality, intelligibility, old-voice retention and adaptation compute.
