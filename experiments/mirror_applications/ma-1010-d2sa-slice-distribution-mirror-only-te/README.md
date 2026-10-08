# MA-1010 — D2SA slice-distribution Mirror code

Status: **NOT ESTABLISHED — blocked before protocol freeze**  
Branch: `research/ma-1010-d2sa-mirror-scan-code-20261008`  
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Draw 14 selected MA-1010 uniformly from 554 eligible P0/UNTESTED candidates. The ordered pool, exclusions, cryptographic seed and replayable index are under `source/`.

## H — Registered hypothesis (not frozen)

The registry proposes replacing part of per-slice MRI test-time adaptation state with a compact scan/slice Mirror coordinate while preserving reconstruction quality and physical data consistency. This hypothesis was not converted into a runnable protocol because its required D2SA native control was unavailable.

## T — Feasibility review

Read PA305 and the D2SA arXiv source. D2SA has two stages: patient-level MR-INR distribution adaptation and per-slice refinement with a learnable anisotropic-diffusion module and frozen convolutional layers. The paper's described experiments use pretrained U-Net or VarNet, undersampled MRI, patient/scanner splits and self-supervised adaptation.

The repository has no D2SA implementation, MRI reconstruction checkpoint, or MRI data. The arXiv source bundle contains no code/repository URL. The current runtime has CPU-only PyTorch (`torch.cuda.is_available() == False`). The thirteenth-sweep research note suggests synthetic phantoms for a first falsification, but phantoms would not reproduce the registered D2SA baseline or its pretrained reconstruction state.

No MRI data were downloaded or inspected; no protocol was frozen; no model or metric was run.

## D — NOT ESTABLISHED

This is a pre-protocol blocker record, not a negative result about Mirror. The experiment cannot compare against the required native D2SA method with the assets currently available. The registry remains **UNTESTED**.

## C — Strongest counter-hypothesis

A faithful small D2SA implementation could be rebuilt from the paper and paired with a synthetic forward model. If so, the absence of public weights/data may not be a fundamental blocker. However, a newly approximated D2SA implementation would not establish reproduction of its pretrained MRI baseline, and comparing Mirror to it would weaken the required control.

## U — Unknown

Whether scan-level variation is compressible into a small Mirror code beyond D2SA's MR-INR and slice-refinement parameters; quality, data consistency, actual payload bytes, adaptation compute, runtime, and fresh-patient behavior are all unmeasured.

## Fact / Interpretation / Hypothesis

- **Fact:** PA305 describes MR-INR patient adaptation plus per-slice learnable anisotropic diffusion; this checkout lacks its implementation, pretrained reconstructor, and MRI data; no protocol or training run occurred.
- **Interpretation:** Without the native control, a Mirror-specific claim is not established.
- **Hypothesis:** A small scan/slice coordinate may still replace part of the adaptation state when tested against a faithful D2SA implementation.
