# MA-948 — scene-cut private residual birth

## H — hypothesis

A predeclared scene-cut signal can spawn private residual state only around cuts, preserving cut-window quality with fewer bytes than always-private per-scene models and fewer errors than a shared-only view.

## T — work completed

Reviewed MA-948’s PA268/PA274 design. Available HNeRV Bunny asset is a single 132-frame scene and has no verified scene cuts; NerVast and other adaptive shared/private INR checkpoints are absent. CUDA is unavailable. No synthetic cut was made and no model metrics were run.

## D — UNTESTED

Without genuine multi-scene cut boundaries and the native shared/private adaptive control, the private-birth decision cannot be evaluated.

## C — strongest counter-hypothesis

A fabricated cut or adjacent temporal windows could test whether code allocation reacts to a discontinuity, but would not show real scene-cut coding or compare to native adaptive INR.

## U — unconfirmed

Cut detection accuracy, private-patch bytes, scene-cut PSNR, temporal error, total RD and decode speed.
