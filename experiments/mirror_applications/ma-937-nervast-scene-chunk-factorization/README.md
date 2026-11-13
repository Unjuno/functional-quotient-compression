# MA-937 — scene × chunk factorization

## H — hypothesis

A factorized scene identity × temporal chunk code can represent unseen scene-window combinations with fewer actual decoder-plus-code bytes while preserving held-out frame quality, scene-cut behavior and decode FPS versus native HNeRV and flat chunk codes.

## T — resource inspection

The official HNeRV repository was cloned at pinned commit `4872129c8d004a25477e0c1ffbbff4ba71943ad5`. Its supplied data/checkpoint cover one 132-frame Bunny video. A CPU decode smoke reconstructed a finite 640×1280 frame from the official quantized embedding and TorchScript decoder (combined source checkpoint bytes 7,574,217 B). This proves the single-video decoder path runs; it does not create a second scene or a held-out scene-window combination.

NerVast native masks/checkpoints are unavailable and the workspace has no CUDA/NVIDIA device. The registered scene×chunk comparison cannot be judged with Bunny time windows relabeled as scenes, so no experiment metrics were generated.

## D — UNTESTED

The exact hypothesis needs multiple fitted scenes and scene-window holdouts. The upstream repository supplies only one video checkpoint. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md), [source/UPSTREAM.md](source/UPSTREAM.md), and [VERIFICATION.json](VERIFICATION.json).

## C — strongest counter-hypothesis

One could evaluate temporal chunk factorization within Bunny on CPU, but the scene coordinate would be constant and that would not test cross-scene generalization.

## U — unconfirmed

Scene×chunk held-out quality, actual multi-video bitstream bytes, scene-cut retention, and decode FPS remain unmeasured.
