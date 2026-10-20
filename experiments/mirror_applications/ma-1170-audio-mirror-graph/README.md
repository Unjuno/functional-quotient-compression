# MA-1170 — Real-Audio Feature Mirror/Loop Graph Autoencoder

Status: **COMPLETED FAIL** under the frozen three-seed protocol. **It is a lossy Mel-feature AE, not a PCM-waveform encoder/decoder.**

## Research insertion

A single 64-number learned virtual graph program generates all 16 directed logical 4×4 edges across four nodes, with two synchronous recurrence steps (Loop) between a 64-band Mel feature encoder and 16-dimensional 8-bit sparse code decoder. **Mirror** creates each shared-core edge weight by Householder→tanh→inverse Householder; the strongest matched **affine** control uses exactly the same learned addresses, 4 pages, gates, and 2 Loop steps without that nonlinearity. The graph is fully active in training; half-edge masking is only a post-training causal diagnostic. Thus, this experiment does not exhaustively search every discrete topology.

## Inputs

Packaged audio dependency: `pyannote/audio/sample/sample.wav` from the referenced environment, 30 sec, 16 kHz, mono PCM16. Its SHA-256 is `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e`. The raw source WAV is **not redistributed**; source/audio.py currently pins the absolute source path in the experimental CPU environment.

Train [0,17) s, dev [18,21) s, **fresh [22,25)** s, late [26,29) s. Per-band mean/std fit from train only. Clip identity shared across splits, so there is no claim of independent-recording generalization. Separate 1.006s Gradio audio file provides only a cross-record diagnostic.

## Run (two-stage, sealed fresh until training completes)

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python source/run.py train --output .
python source/run.py audit --output .
python source/ablate.py .       # posthoc only; never changes H1/H2 preregistered gates
python source/pca_control.py  # posthoc strong linear control
python source/check_flac.py  # separate exact PCM16/FLAC reference
```

Training creates 12 CPU FP32 model checkpoints and `results/TRAIN_DEV_ONLY.csv` with no fresh evaluation. The audit refuses to run until all 12 checkpoint files and the training verification manifest exist. The separately frozen protocol is `PROTOCOL.json` (SHA-256 `86da7956cfac5eab05eb93be4b15fabeb3c9f46cab4807eb7edf1f3ae431557f`). The earlier research branch protocol *summary* contains a historical JSON SHA not present as a raw file; that historical SHA is **not independently verifiable**.

## Results

See `REPORT_JA.md`, `RESULTS_CORE.csv`, `VERIFICATION.json` and all checkpoints. Fresh NMSE 3-seed medians: **mirror 0.342928**, **affine 0.333077**, **sparse 0.347384**, **native 0.340326**, and posthoc **rank16 PCA 0.229338**. `pytest` **9/9** and all packet-replay checks PASS. Main Mirror-specific quality/localization gates FAIL.

The 16×8-bit latent has 128 bit/frame (~12.8 kb/s for 10ms hop), but **no waveform reconstruction exists** and the code is not lossless. All code streams + model packet bytes include the actually serialized learned state and 16-scalar scale, excluding the runtime implementation itself. See `results/LOSSLESS_FLAC_REFERENCE.json` for an actual exact audio roundtrip using FLAC.

## Prior art

- SoundStream (Zeghidour et al., DOI 10.1109/TASLP.2021.3129994)
- EnCodec (Défossez et al., arXiv:2210.13438)

These are **not** reproduced; the experiment is not a production audio-codec benchmark.
