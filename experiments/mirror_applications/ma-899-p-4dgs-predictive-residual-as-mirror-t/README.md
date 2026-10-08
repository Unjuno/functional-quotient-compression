# MA-899 — P-4DGS predictive residual as Mirror temporal code

Status: **NOT ESTABLISHED — blocked before protocol freeze**  
Branch: `research/ma-899-p4dgs-mirror-predictive-residual-20261008`  
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Draw 16 selected MA-899 uniformly from 551 eligible P0/UNTESTED candidates. The ordered pool, exclusions, seed, replayable index and hash are under `source/`.

## H — Registered hypothesis (not frozen)

The registry asks whether temporal Mirror prediction residuals can improve the rate-distortion frontier of a shared spatio-temporal Gaussian anchor predictor beyond P-4DGS predictive entropy coding, while retaining frame access and rendering speed. No experiment protocol was frozen.

## T — Feasibility review

PA251 describes P-4DGS spatial-temporal anchor prediction with context-adaptive entropy coding. The phase-two research note requires native 4DGS/ADC-GS/CC-4DGS/P-4DGS, independent scene or motion assets, actual coded bitstream bytes, random-access measurements, and measured frame FPS on named hardware.

The P-4DGS arXiv source bundle contained no code/repository URL. This checkout has no P-4DGS codec, dynamic Gaussian scene dataset, or compatible rendering assets. PyTorch is CPU-only (`torch.cuda.is_available() == False`); the note calls for hardware-accelerated rendering. No scene/video data were downloaded or inspected, and no protocol or model run was started.

## D — NOT ESTABLISHED

This is a prerequisite blocker, not a negative Mirror result. Without the predictive entropy codec and a renderer/data path, a CPU toy would omit the defining native control and could not answer the registered rate-distortion/FPS question. Registry remains **UNTESTED**.

## C — Strongest counter-hypothesis

A faithful software renderer and a public P-4DGS code/data release may allow a smaller CPU-only diagnostic. It would still not establish the registered hardware FPS frontier without the required accelerated renderer.

## U — Unknown

Whether a low-description Mirror temporal residual can beat ordinary predictive residual entropy coding at equal rendered quality, coded bytes and random-access latency remains untested.

## Fact / Interpretation / Hypothesis

- **Fact:** PA251 makes predictive entropy coding the native control; local implementation, scene assets and accelerated renderer are absent; no data or metrics were accessed.
- **Interpretation:** The required codec/storage/runtime comparison is not reproducible in this environment.
- **Hypothesis:** Mirror temporal residuals may add value only where they beat ordinary predictive coding at matched distortion and actual bitstream size.
