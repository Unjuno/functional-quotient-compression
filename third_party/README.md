# Third-party code

Third-party snapshots used as experimental baselines live here. Preserve their original license and keep project-specific modifications outside the vendor directory whenever practical.

## nanoGPT

`nanoGPT/` is a runnable research baseline derived from a source archive supplied by the project user on 2026-10-07. See `nanoGPT/VENDOR_NOTES.md` for exact archive provenance and the committed subset.

- original archive SHA-256: `7f8b869f501ef8015407399e92f895d2bc6113f64fb8444f67de6b010febd992`
- upstream license in snapshot: MIT, copyright Andrej Karpathy
- purpose here: small, readable GPT baseline for systematic Mirror-application experiments
- upstream README in the supplied snapshot says nanoGPT is deprecated in favor of nanochat as of Nov 2025; this project intentionally uses nanoGPT as a compact hackable baseline, not as a claim that it is the newest training stack.