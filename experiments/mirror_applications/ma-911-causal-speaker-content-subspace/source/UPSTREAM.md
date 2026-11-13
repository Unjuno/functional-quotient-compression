# MA-911 upstream and data review

- Official repository: `mjukus/interventional-post-training-speech`, commit `140545059e7f4e81cc02efbacd499ec56b62540e` (clone at `/tmp/ma911-interventional-speech`, reviewed 2026-11-13).
- PA257: Cox & Barker, arXiv:2606.17967 v2; official repo was located. The paper uses interventional contrastive training to separate speech content and speaker factors, evaluating OOD speaker verification and keyword spotting.
- Official README requires an F5-TTS Apptainer image to generate interventional audio, LibriTTS test-clean for source speech, precomputed speech-foundation-model features for training, SpeechCommands for content evaluation and VoxCeleb1 test audio for speaker verification. The repo only includes `data/VoxCeleb1/gender.txt`; it does not include the audio, synthetic set, features or trained checkpoints.
- Official synthetic feature generation uses a pretrained SSL speech model and the generated intervention corpus. Dataset samples combine reference speaker utterances with target content rendered by F5-TTS. No VoxCeleb, LibriTTS, SpeechCommands or synthetic wav/feature files are present in `/workspace`.
- The official project supports CPU PyTorch, but current runtime has PyTorch 2.14.1+cpu, no `torchaudio`, no Apptainer binary, and no CUDA device. Even if CPU training were possible, required audio/features are absent and creating the intervention set requires F5-TTS infrastructure.
- Zenodo records 20735589 and 23417942 contain only source archives (~276 KB), no dataset/checkpoint assets.

This is sufficient to keep the registered audio experiment UNTESTED. A handcrafted latent matrix or ordinary same-speaker speech pair would not establish causal speech-content separation.


Astra execution advice (2026-11-13) independently confirmed the same boundary: keep MA-911 UNTESTED and advance, because a tiny analytic substitute changes the hypothesis and cannot satisfy speaker/OOD checks.
