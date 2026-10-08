# MA-186 — continual task acquisition through Mirror codes

## H — falsifiable hypothesis

After common Task 0 pretraining, a frozen shared linear backbone plus one learned Givens angle per new task will retain earlier aligned skills with no more than 1.10× rank-2 LoRA's final mean task MSE, while using at most 0.25× LoRA's actual incremental inference bytes per skill. Unrelated task maps should expose the point where private parameters are needed.

## T — protocol

Five known task IDs map 16D inputs to 8D outputs. Task 0 receives 600 AdamW updates; each of four later tasks receives 300 updates on 64-example batches. Aligned teachers use different Givens rotations of one common task map. The unrelated condition uses independent task matrices.

Controls are sequential full-weight fine-tuning, per-task rank-2 LoRA, shared hypernetwork-generated rank-2 adapters, hard tying, and independent full task weights. Development seeds were 18601–18602 at LR .003 and .01; the preregistered mean final MSE across methods and both task families selected .01 (.57687 versus .58055). Fresh seeds 18611–18613 used only the frozen .01 setting.

Inference bytes use a deterministic binary record format charging field names, dtype/shape metadata, dimensions, method, IDs and tensor bytes. Resume payloads use `torch.save` and include inference state plus optimizer state. Matched-format baselines are used for incremental byte calculations (amendment A2). The initial `torch.save` inference screen and first A1 screen with mismatched baseline formats are retained as diagnostics, not used for selection or final metrics.

## D — decision

**FAIL for the registered quality gate; storage and retention subclaims pass in this synthetic aligned task.** This is fixed-update evidence, not a capacity result.

Fresh aligned results:

| Method | Mean final seen-task MSE | Total inference payload | Incremental bytes/skill | Total resume payload | Mean inference examples/s |
|---|---:|---:|---:|---:|---:|
| Mirror code | 5.92e-11 | 707 B | 20 B | 7,082 B | 2.26M |
| Rank-2 LoRA | 4.56e-11 | 1,590 B | 241 B | 13,178 B | 6.54M |
| Hard tie | 3.45e-3 | 628 B | 1 B | 2,657 B | 15.98M |
| Hypernetwork rank-2 | 3.45e-3 | 2,610 B | 495 B avg | 11,510 B | 1.72M |
| Independent full | 6.51e-7 | 2,988 B | 589 B | 17,210 B | 16.63M |

The Mirror/LoRA final-MSE ratios on fresh aligned seeds were 1.85, 1.23 and 1.05. The quality requirement failed on two of three worlds. Mirror passed the per-skill byte limit on all three, used 55.5% fewer total inference bytes than LoRA, and had zero measured increase in earlier-task MSE after adding later aligned tasks. Its mean measured inference throughput was 0.35× LoRA. Resume payload was 46.3% smaller overall, though its incremental resume state remained 1,106 B/skill versus 2,630 B/skill for LoRA.

For unrelated task maps, the Mirror view could not recover the independent functions (fresh final mean seen-task MSE roughly 1.59); independent full weights reached roughly 1e-7 at much greater bytes. This locates a private-parameter boundary for this one-angle view. Rank-2 LoRA also remained imperfect on unrelated tasks.

## Fact / interpretation / hypothesis

**Facts.** The 20 B versus 241 B increment includes the record name, scalar float32 code, task identifier accounting, and codec overhead. The serialized fresh payloads were 707 B versus 1,590 B. Mirror missed the registered relative-quality threshold in two fresh worlds. Earlier-task MSE did not rise in any of the three aligned fresh worlds. Runtime was lower than LoRA throughput.

**Interpretation.** A one-angle input view can encode the aligned task orbit with a strong inference-storage reduction, but the fixed training recipe does not reach LoRA quality reliably. The unrelated task condition requires more expressive private state. This does not establish a Mirror-specific advantage over hypernetworks: the implemented hypernetwork control stayed at the hard-tie result because both zero-initialized low-rank factors yield zero gradients at initialization, so this control was degenerate and is not a fair learned-hypernetwork comparison.

**Hypothesis.** A non-degenerate low-rank hypernetwork initialization or a task-specific multi-angle view may close the aligned quality gap, but either may consume more compute or bytes. Testing either after seeing fresh results requires a new registered experiment/amendment and new fresh worlds.

## C — strongest counter-hypothesis

The quality miss may be caused by the single scalar view and optimization conditioning rather than a general limit of shared Mirror views. Conversely, rank-2 LoRA's 1.10× advantage is small in absolute MSE near the noise floor, so the relative threshold is sensitive; the registered criterion is nevertheless applied as written.

## U — unresolved

- The shared hypernetwork control is degenerate; no Mirror-specific superiority claim is supported.
- No optimized kernel was tested; measured throughput comes from the eager PyTorch implementation and tiny CPU harness.
- Only known task IDs and synthetic linear maps were tested; no task discovery, language model, or natural continual-learning data.
- The fixed update schedule does not establish converged capacity.

## Verification

`PROTOCOL.json`, raw development/fresh result tables, source, tests, and `VERIFICATION.json` preserve the preregistration, deviations, and replay evidence.
