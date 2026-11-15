# MA-1031 — SAM2 interactive correction prompt views

**Decision: FAIL — scoped to three synthetic SAM2.1 Hiera-tiny CPU videos.**

## H — Hypothesis

A positive/negative click pair can be represented by a shared per-object radius and per-frame midpoint/rotation View, keeping corrected J&F within 1 pp of direct human-like clicks while saving at least 25% of serialized correction-history bytes. The test also checks whether this state reduces click count; it was not preregistered to do so.

## T — Run

- Official SAM2.1 Hiera-tiny, 256px CPU, source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`; checkpoint hash in `VERIFICATION.json`.
- Four prompted object tracks, six frames per video. An exact mask prompt initializes each object on frame 0. At each later current frame, the correction simulator chooses one positive click from false negatives and one negative click from false positives, using that frame’s current prediction and ground truth only. Future frames/masks are never consulted.
- Development seed 103100 set a shared half-radius per object. This setting was committed before audit videos were generated. Fresh audit seeds 103101–103103.
- Controls: zero correction, direct float32 clicks, native 8px-grid click coding, Mirror polar pair code, and full-mask correction upper control. All SAM2 outputs and cached frame state are included in the measured resumable session bundle. Large serialized model-state binaries were not retained; their exact in-memory lengths and SHA-256 values are recorded. Compact correction-history payloads, inputs, and prediction masks are retained.

## D — FAIL

### Fact

| Method | Mean J&F | Clicks/video | Correction-history bytes | Full-session bytes | Mean FPS | ID switches by seed |
|---|---:|---:|---:|---:|---:|---|
| No corrections | 0.568154 | 0 | 119 | 169,981,586 | 3.173 | 0, 0, 2 |
| Direct clicks | 0.833447 | 40 | 544 | 170,346,906 | 2.162 | 0, 0, 0 |
| 8px-grid clicks | 0.838396 | 40 | 318 | 170,346,650 | 2.145 | 0, 0, 4 |
| Mirror polar pair | 0.773317 | 40 | 503 | 170,346,863 | 2.185 | 0, 0, 0 |
| Full-mask upper control | 0.949184 | 0 | 121* | 175,595,954 | 2.812 | 0, 0, 0 |

`*` The full-mask correction maps are stored in the SAM2 mask-input state and charged in the full-session bytes, not repeated in the compact click-history archive.

Mirror history is 7.5% smaller than direct float32 click history (503 vs 544 B), below the frozen 25% gate. It uses the same 40 clicks/video as direct interaction; no correction clicks were removed. Mirror’s mean J&F is 6.0 pp below direct clicks and 6.5 pp below the 8px-grid control. The grid control is 185 B smaller in correction history and 213 B smaller for the full session while scoring higher than Mirror. Mirror’s full-session savings versus direct are only 43 B, about 0.000025%. It is slightly faster than both direct and grid coding, but that does not offset its quality loss and larger payload than the grid control.

Direct corrections improve mean J&F by 26.5 pp over no correction. The task is therefore responsive to interaction prompts; the Mirror representation fails to preserve that benefit well enough.

### Interpretation

The click-pair radius is not stable across frames. Holding it shared while encoding midpoint and angle moves the reconstructed clicks enough to lose segmentation quality. A simpler coarse coordinate grid is both smaller and more accurate in this test. Even the Mirror’s 41 B prompt-history reduction nearly disappears once the actual SAM2 session and common model state are charged.

### Counter-hypothesis (C)

The 8px grid’s success may reflect SAM2’s tolerance to small click displacement on these synthetic videos, and its coordinates are already confined to a 256px image. This is not evidence that low-bit click coding transfers to natural interactive segmentation.

### Unknown (U)

Natural videos, DAVIS/YouTubeVOS, longer interactions, varied click policies, higher-resolution input, GPU throughput, and learned or temporal prompt bases remain untested. The screen encodes prompt history but does not reduce the number of human corrections.

## Verification

The shared per-object radii were frozen in commit `3c6fa6cd3e9cb2e503ae83a62f1e76cc9bde9fd5` before audit data generation. Fresh input, history and prediction hashes pass; J&F and ID-switch metrics replay from saved predictions; full-session byte arithmetic verifies. Four tests pass. Full session archive hashes and exact byte counts were computed from serialized state but the large state byte strings were not retained in Git; their hashes are retained in `runs/audit_rows.jsonl`.
