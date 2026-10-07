# nanoGPT vendor notes

Source: user-supplied `nanoGPT.zip`, inspected 2026-10-07.

Original ZIP SHA-256:
`7f8b869f501ef8015407399e92f895d2bc6113f64fb8444f67de6b010febd992`

Source-only snapshot hash (all original files except two images and two notebooks):
`5d1eccdbb3fdc28d4979570a224ca414edb6da1e9a1a708d875f193adcec36aa`

License: MIT; see `LICENSE`.

This vendor copy is intentionally used as a stable experimental baseline. Project-specific Mirror variants should be implemented in experiment directories or thin adapters rather than silently changing the vendor copy.

Not committed from the supplied archive because they are not required for execution:
- `assets/nanogpt.jpg`
- `assets/gpt2_124M_loss.png`
- `scaling_laws.ipynb`
- `transformer_sizing.ipynb`

The supplied snapshot README states that nanoGPT is old/deprecated and points to nanochat. We retain nanoGPT because its compact model/training implementation is useful for controlled architecture modifications.

Inspection found no executable binaries, shared libraries, DLLs, or shell scripts in the supplied archive.