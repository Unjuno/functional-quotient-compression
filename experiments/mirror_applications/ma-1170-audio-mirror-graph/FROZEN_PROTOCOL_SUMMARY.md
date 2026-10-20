# MA-1170 — Frozen acoustic feature encoder experiment

Date: 2026-10-20 JST. **Frozen before running any model training.** Local complete JSON protocol SHA-256: `bc14a69de4ee2afd1ba69b404aabe7018c1df3ea4d0d07f112ca7f5f001ffcbb`.

- Insertion: one 64-float learned virtual graph program (16 shared core, 16 gates, 32 differentiable addresses), yielding 16 4×4 logical edge matrices simultaneously; four nodes × two recurrent Loop passes between sparse audio encoder and decoder. Fixed four signed-permutation pages; nonlinear Householder–tanh–inverse versus same-address affine control.
- Audio: the bundled 30 s, 16 kHz PCM mono `pyannote.audio/sample/sample.wav`, SHA-256 `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e`. Data is not copied into GitHub.
- Preprocessing: log1p Mel power, 64 bands, FFT400, hop160, train-only per-band mean/std. Nonoverlapping time windows: train [0,17), dev [18,21), sealed fresh [22,25), later temporal holdout [26,29). Diagnostic cross-record: packaged Gradio 8 kHz 1.006-second audio, resample 16 kHz.
- Four competitors: `mirror`, `affine`, `sparse` (TopK=6, 16 latent), `native` (direct hidden-layer TopK=6). All 16-latent and 8-bit quantized via train-only per-dimension 99.5th percentile scales. Respective parameter budgets: 2144,2144,2080,2352.
- Torch FP32 CPU one thread; seeds [31,37,41]; 500 AdamW updates per model, LR=.003, batch=96, weight decay .0001, fixed paired sampling, dev check each 50 steps; best dev checkpoint. Gate L1 coeff .0005 for graph models. No fresh tuning.
- Primary: unquantized fresh feature NMSE. Secondary: quantized fresh NMSE, later temporal holdout, cross-record, actual model packet bytes, quantized latent bytes, selected Mel-band localization from per-latent ablation, masking bottom 8/16 edges, isolated CPU B1/B64 inference latency.
- Frozen gates: Mirror quality PASS only if 5% fresh NMSE improvement over affine on ALL three paired seeds and median better than direct sparse AE. Localization PASS only if +0.05 median 8-band ablation concentration above both affine and sparse in ALL seeds without >5% quality regression. No generalization claim from one file.
- Absolutely not: a raw audio codec, exact lossless reconstruction, entropy-optimal compression, a general speech model, SoundStream/EnCodec reproduction, or a capacity proof. File/program bytes and inference buffers need independent accounting.

Prerequisite: protocol first, then source, tests, model training, fresh opening, forensic report. Reports preserve all failure outcomes.
