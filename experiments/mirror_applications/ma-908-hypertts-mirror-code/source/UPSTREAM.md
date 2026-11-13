# HyperTTS provenance and execution review

- Official source: `declare-lab/hypertts`, commit `2e2b1a1d284849a903d41c005bb32dff9202a158` (cloned 2026-11-13 to `/tmp/ma908-hypertts`; clone is external to this experiment package).
- README links pretrained LTS100 `600000.pth.tar` through Google Drive. Downloaded to `/tmp/ma908-600000.pth.tar`, 436,038,607 bytes (SHA-256 `f99fa642068a6500303fe6a1ab9ca5fc687f34b58090bae73a398b922cf4d199`). The model artifact is not committed.
- Repo contains VCTK/LJSpeech preprocessing manifests and 24 example synthesized WAVs, but not raw utterance audio, VCTK reference audio, or required speaker embedding `.npy` arrays. `preprocessed_data` is metadata/manifests, not the speech corpus.
- README documents CUDA-based LTS/VCTK adaptation and a 900k VCTK synthesis setup. Current runtime is PyTorch 2.14.1+cpu. Repo requirements pin torch 1.7.0, TensorFlow 2.5.1, NumPy 1.19.4 and older speech packages. A first import in current Python 3.12 failed on missing TF-linked DeepSpeaker requirements after several compatible packages were installed.
- Without held-out reference speech and speaker embeddings, the registered speaker quality/HyperTTS-vs-Mirror comparison cannot be run. Random speaker vectors or unrelated TTS would not test the registered hypothesis.
