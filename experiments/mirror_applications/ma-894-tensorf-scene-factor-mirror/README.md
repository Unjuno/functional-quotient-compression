# MA-894 — TensoRF scene × component Mirror factorization

Status: SCREENING
Evidence lane: SYNTHETIC NEURAL GRAPHICS / MECHANISM / STORAGE / COMPUTE
Base commit: `c738b4b1e99ce86e12157c963232138663e395e2`

## H — hypothesis

When several fields reuse separable spatial atoms, a compact scene code over shared tensor components can reconstruct fresh fields from partial coordinate samples with substantially fewer actual serialized bytes than independent dense fields.

## Scope

This is a synthetic scalar-field mechanism screen. It uses a rank-4 scene × X × Y × Z CP representation and compares that Mirror naming of scene coefficients with the algebraically identical native scene-mode CP control and a flattened-field PCA control. The registered real-scene TensoRF result remains unestablished.

The official TensoRF implementation (PA246) is PyTorch based and its core tensor factors use `grid_sample`; its README lists ordinary Python packages and synthetic Blender scenes. No custom CUDA extension is required by the core factorization. No official scene dataset is currently present, so this screen does not train or render official TensoRF scenes.

## Controls

1. Independent dense field payload per scene.
2. Native scene-mode CP factorization with the same scene axis and spatial axes.
3. PCA over flattened development fields, with fresh scene codes fitted from the same observed coordinates.
4. Mirror-coded shared spatial factors.

## Frozen protocol

See `PROTOCOL.json`. Development uses seeds 89401 and 89407. Fresh seeds are 89501, 89503, and 89509, with six development scenes, three fresh scenes, rank four, and half of each fresh field's coordinates used to fit its code.

Actual NPZ serialized payload bytes are authoritative. Query and fit costs are measured separately. Fresh seeds are not used to tune the method.

Post-run scope corrections are recorded in `PROTOCOL_AMENDMENT_01.md`: only fresh inference codes are included in deployment bytes, and the query wall-clock is invalid for cross-method comparison.

## Claim boundary

Even a positive storage result would be a synthetic shared-subspace signal. A match by native scene-mode CP or PCA is evidence against Mirror-specific value. No novel-view rendering, real-scene PSNR, or FPS claim follows from this experiment.
