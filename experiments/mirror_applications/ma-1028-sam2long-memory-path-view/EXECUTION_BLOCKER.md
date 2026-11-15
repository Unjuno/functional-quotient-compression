# MA-1028 execution blocker

## Decision

**NOT ESTABLISHED** for the registered two-object test. The one-object run is preserved as an exploratory implementation diagnostic only.

## Evidence

1. The frozen `PROTOCOL.json` requires two objects and three SAM2Long pathways.
2. The runner that completed a full clip used one object. The official SAM2Long path-selection code reads `object_score_logits.item()` while ranking a path (`sam2/sam2_video_predictor.py`, inside `_run_single_frame_inference`) and has an adjacent source comment that `num_object` is “always one.” With two objects, `.item()` receives more than one value. This prevents reproducing the registered multi-object tree control through the official implementation.
3. The official `tools/vos_inference.py` submits every object mask into one inference state, then invokes one tree propagation. Its path-selection code is not vectorized over object IDs.
4. A 256px feasibility run stopped before scoring because the relative attention code hardcodes 4096 spatial tokens. A pre-score amendment to official 1024px succeeded for a three-frame smoke and for the exploratory one-object 12-frame run.

## Exploratory one-object data (not registered evidence)

Across independent, hard-shared, Givens Mirror, and rank-1 additive, the selected masks were byte-identical on seed 102801: J&F 0.844057, J 0.873851, boundary F 0.814263. Path memory bytes were 16,777,216 / 6,291,456 / 6,291,696 / 11,534,396 respectively. Full-session bytes were 337,462,081 / 326,977,431 / 326,979,305 / 332,223,578. The official model, config, normalized image cache, predictions and metadata are charged. These results do not establish two-object quality or generalization.

For this one-object payload shape, path memory is only 4.97% of complete-session bytes; even removing it entirely cannot meet the 25% complete-session storage gate. Mirror is 240 B larger than hard sharing and gives the same selected masks. Timing is a single CPU run and is descriptive only.

## Next action

The registered test needs an official tree implementation that scores multiple objects without scalar `.item()` assumptions. Preserve MA-1028 as NOT ESTABLISHED and continue with the next P0 candidate; do not reinterpret the one-object screen as a two-object result.
