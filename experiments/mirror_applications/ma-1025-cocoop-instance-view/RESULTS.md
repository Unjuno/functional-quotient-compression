# MA-1025 — CoCoOp image-conditioned Mirror context

Status: **FAIL** for the preregistered Mirror-specific gate. This is a small, bounded CPU screen on CIFAR-10 and frozen CLIP, not a broad vision-language result.

## H — Hypothesis

A two-angle image-conditioned Givens code applied to one shared four-token CLIP context will retain CoCoOp-style base-to-novel transfer, use fewer learned-state bytes than a full generated 512D context bias, and improve novel-class accuracy by at least 1 percentage point over a matched rank-2 additive prompt conditioner.

## T — Trial

- Frozen `openai/clip-vit-base-patch32` (151,277,313 parameters; 608,863,580 B for all recorded weight/config/tokenizer/processor files), CPU inference/training.
- CIFAR-10 classes 0–5 (`airplane` through `dog`) were training classes; classes 6–9 (`frog`, `horse`, `ship`, `truck`) were held out from training.
- 4 training images per base class (24 total); 4 official test images per class (40 total). Selection seed 102501. Training and test indices were explicitly excluded from MA-1024's corresponding index sets; image hashes are recorded in `runs/development.json`.
- Learned states were frozen before the official test images were opened. Every learned method received 20 Adam updates, batch size 4, learning rate 0.001, four context tokens, and a hidden width of 32.
- Controls: zero-shot CLIP, shared CoOp, full CoCoOp-style image-conditioned 512D context bias, two-angle Givens Mirror, and an equal-width two-value additive prompt code with a learned rank-2 basis.
- CoCoOp-style conditioning was implemented as a shared image-feature MLP (512→32→512) added to each context token. Mirror uses (512→32→2) and two paired Givens rotations. The additive control uses the same (512→32→2) code MLP and a learned 2×4×512 prompt basis.
- Actual prompt payloads use a UTF-8 JSON tensor manifest followed by contiguous little-endian float32 state arrays. Full-bundle bytes add the complete frozen CLIP asset files. The harness regenerates per-image conditional text features and also recomputes shared prompts batch-by-batch. The reported per-image wall is the full 40-image evaluation wall divided by 40, so it is an unoptimized diagnostic, not a serving benchmark.

## D — Decision

After training, explicit inference-operation, token-template, feature-normalization and logit-scale metadata were added to each serialized payload. Payloads were regenerated from the unchanged learned tensors, all byte totals below include the added metadata, and serialized-state replay was rerun. `runs/development.json` preserves the original and amended payload hashes and byte deltas.

**FAIL** under the frozen gate. Mirror satisfies the CoCoOp accuracy tolerance and has a smaller learned state than full CoCoOp, but it does not beat the matched rank-2 additive control. Both had exactly the same recorded base, novel, and harmonic-mean accuracy. Therefore no Mirror-specific quality advantage is established.

### Facts

| Method | Base acc. | Novel acc. | H-mean | Learned payload | Full bundle | Eval wall / image |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 83.33% | 81.25% | 82.28% | 326 B | 608,863,906 B | 0.0921 s |
| Shared CoOp | 75.00% | 81.25% | 78.00% | 8,702 B | 608,872,282 B | 0.7235 s |
| CoCoOp-style full bias | 70.83% | 75.00% | 72.86% | 142,236 B | 609,005,816 B | 0.3081 s |
| Mirror Givens | 83.33% | 81.25% | 82.28% | 74,935 B | 608,938,515 B | 0.2665 s |
| Additive rank-2 | 83.33% | 81.25% | 82.28% | 91,390 B | 608,954,970 B | 0.2753 s |

- Novel group has only 16 examples: the reported 6.25 pp Mirror-over-CoCoOp difference is one image. Mirror and additive have identical predictions' aggregate accuracy at this audit resolution; this does not imply their per-image predictions are identical.
- Mirror learned-state payload is 47.3% smaller than full CoCoOp-style state and 18.0% smaller than the additive rank-2 state.
- Mirror's complete bundle exceeds the frozen backbone by 74,935 B, a 0.0123% delta. The full-bundle difference from additive is 16,455 B (0.0027% of the backbone).
- Training wall time was 31.44 s (shared CoOp), 36.13 s (CoCoOp-style), 54.66 s (Mirror), and 49.45 s (additive). Feature extraction took 1.18 s for training examples and 1.86 s for audit images. Total run time was 246.05 s. All timings are CPU-only and single-run.
- Test inference wall for all 40 images was 3.68 s, 28.94 s, 12.32 s, 10.66 s, and 11.01 s respectively. There is no claim of GPU throughput.
- First-versus-last sampled minibatch training losses were 0.318→0.529 (shared CoOp), 0.746→0.125 (CoCoOp-style), 0.315→0.702 (Mirror), and 0.321→0.337 (additive). These are two minibatches from a shuffled 20-update run, not convergence curves; their non-monotonicity makes this a fixed-update screen only.
- Exact serialized payload replay reproduced all five methods' base, novel and harmonic-mean metrics. Backbone assets, serialized payloads, audit IDs and audit image hashes were checked.

### Interpretations

- This screen shows that a compact image-conditioned code can score well on this small split and use fewer prompt-state bytes than the full CoCoOp-style generator.
- Its measured advantage over the full generated bias is confounded by the rank-2 additive control, which exactly matches its aggregate accuracy. The study does not establish that Givens geometry is useful.
- Because the backbone is over 608 MB, prompt-generator compression barely moves complete inference storage. Shared CoOp is also smaller and matches Mirror's novel accuracy here.
- The apparent CoCoOp deficit is not evidence against the CoCoOp method generally: the training sample (24 images), 20-update budget and 16 novel audit images are intentionally small.
- The timing harness recomputes text features batch-by-batch for all methods; shared and zero-shot prompts could be cached in an optimized serving implementation. These wall times are implementation-specific diagnostics and do not establish an inference-compute Pareto improvement.

### Hypotheses for follow-up

- The task-specific function may need a higher-rank additive basis or a larger/private context to improve class transfer; these were outside this frozen candidate.
- If conditional prompt coding is revisited, a larger disjoint audit and matched optimization/convergence study should precede any capacity or generalization claim.

## C — Strongest counter-hypothesis

The rank-2 additive conditioner can represent the useful variation in this task just as well as the Givens view; the apparent six-point novel advantage over CoCoOp is one audit image and may be sampling noise. Shared CoOp's equal novel score with much less state further weakens a practical compression claim.

## U — Unconfirmed

- Performance on larger class sets, multiple datasets, multiple data seeds, larger audit sets, or converged training.
- Whether Givens geometry offers any quality, sample-efficiency, or runtime advantage beyond ordinary low-rank conditioning.
- Any broad CoCoOp, CLIP, or VLM claim. No GPU, QPU, multi-domain, or production serving result is claimed.

## Fact / Interpretation / Hypothesis separation

The table, hashes, run counts and replay outcome are **Facts**. The statements about why the rank-2 control matches and about full-bundle practical value are **Interpretations**. Suggestions about rank/budget and future testing are **Hypotheses**.
