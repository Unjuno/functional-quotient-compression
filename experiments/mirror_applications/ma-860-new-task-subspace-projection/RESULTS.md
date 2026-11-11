# MA-860 results

## Fact

- A pooled MLP was trained on three source rotations (0°, 90°, 180°); three source-task fine-tunes from the shared base supplied the rank-3 delta basis. A new 270° task was adapted from 32 training support examples for 100 updates. Development seed 53 selected base/code learning rates (.1, .5, .01 random, .01 full). Fresh support seeds: 223, 239, 257; the fixed test split was disjoint from both base training and support selection.
- Heldout 270° accuracy for learned rank-3 coordinates was .3750, .3944, .4194. The matched random rank-3 control scored .3806, .4861, .4194. Full fine-tuning scored .7861, .7250, .6806. The unadapted base scored .4306, .4861, .4389.
- Actual serialized base+basis+one-code payload was 41,301 B; the full adapted MLP was 11,933 B. The stored basis alone was 30,441 B. Thus one-task use costs 3.46× the full checkpoint. Rank-3 reconstruction proxy is 7,230 MACs per reconstruction; query proxy is 852,480 MACs for 360 examples.
- Native coefficient and Mirror coefficient outputs are identical (same coefficient decode); serialization reload max logit difference was 0. Across each full adaptation pipeline, elapsed time was 0.58–1.58 sec on CPU.

## Interpretation

The source-task delta basis did not transfer to the held-out rotation: it matched or lost to the random basis and fell far below full fine-tuning. The basis payload dominates the small task code and makes the one-task representation much larger than a standalone model. This screen provides no quality or single-task storage benefit for projection.

Mirror coordinates are an identity re-encoding of native coefficients and add no function or quality beyond native subspace optimization. This continues the direct-chart alias finding, while the poor transfer result is specific to this small rotated-Digits task setup.

## Hypothesis

A small basis learned from related source tasks could support few-shot adaptation to a held-out task when the task update lies in the source-delta span. In this rotation screen, it does not.

## H / T / D / C / U

- **H:** A rank-3 learned task basis can adapt to a new task with near-full fine-tune quality and compact storage, compared with random rank-3 and full fine-tune controls.
- **T:** 64→32→10 MLP; source tasks 0°,90°,180°; held-out 270°; 32 support examples; 100 Adam updates; development seed 53 selected learning rates; fresh seeds 223/239/257; random subspace, no-adaptation, and full fine-tune controls. Actual serialized bytes and MAC proxies measured.
- **D:** **FAIL** for quality, storage, and Mirror-specific attribution. Learned rank-3 accuracy was .375–.419 vs .681–.786 full fine-tune, and 41,301 B shared payload vs 11,933 B full model. Native and Mirror coefficient outputs are identical.
- **C:** The task delta directions from only three rotations do not span the 270° adaptation; 32 examples also leave full tuning underfit, but it still substantially outperforms both rank-3 methods.
- **U:** More source tasks, larger bases, natural tasks, multiple held-out tasks amortizing the basis, additional support sizes, and pretrained language models remain untested.
