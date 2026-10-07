# nanoGPT baseline notes

Source: user-supplied `nanoGPT.zip`, inspected 2026-10-07.

Original archive SHA-256:
`7f8b869f501ef8015407399e92f895d2bc6113f64fb8444f67de6b010febd992`

Original extracted source-only tree hash (all archive files except two images and two notebooks):
`5d1eccdbb3fdc28d4979570a224ca414edb6da1e9a1a708d875f193adcec36aa`

License: MIT; see `LICENSE`.

This directory is a **runnable research baseline derived from the supplied snapshot**, not a bit-for-bit mirror of the archive. Comments and nonessential prose were normalized while the core nanoGPT model/training behavior was kept as the baseline target. The original archive hash above is the provenance authority.

Committed core:
- model / train / sample / benchmark / configurator
- GPT-2 and Shakespeare configs
- dataset preparation scripts and dataset notes

Not committed because they are not needed for architecture experiments:
- `assets/nanogpt.jpg`
- `assets/gpt2_124M_loss.png`
- `scaling_laws.ipynb`
- `transformer_sizing.ipynb`

Project-specific Mirror variants should live outside `third_party/nanoGPT/`, so baseline behavior and experimental changes remain separable.

The supplied upstream README says nanoGPT is old/deprecated and points to nanochat as of Nov 2025. This project intentionally keeps nanoGPT as a compact hackable baseline.

Archive inspection found no executable binaries, shared libraries, DLLs, or shell scripts.