# MA-199 — task-specific coordinates in a shared gradient plane

Status: **FAIL at development**; fresh worlds stayed sealed
Branch: `research/ma-199-gradient-coordinate-20261007`  
Protocol frozen at commit `6a2e22c08c3e766359c986e4b7202b4651feee39`

## H — falsifiable hypothesis

When task updates occupy a shared two-dimensional left gradient plane, one task-specific Givens angle and one update vector should retain aligned-task quality with less inference and optimizer-state storage than private rank-1 LoRA. Unrelated directions should require private capacity.

## Prior-art delta

FedLore (arXiv:2610.01620) maintains Adam moments in shared projected gradient coordinates and refreshes the common basis between rounds; it reports optimizer moments of size 2rn. GaLore (arXiv:2403.03507) uses projected gradient coordinates to reduce optimizer state. OGD (arXiv:1910.07104) and iGSP (arXiv:2605.19301) protect prior tasks by projecting gradients against previously stored task subspaces. PA05/PA06 establish generated low-rank adapters as alternatives.

This synthetic screen supplies and charges one shared 16x2 plane P, then compares a task update `(P @ [cos(theta), sin(theta)]) @ v_t.T` against generic `P @ C_t`, LoRA, a shared hypernetwork, and independent weights. It does not claim the random plane could be discovered from natural task gradients at this cost.

## T — protocol and execution

One frozen 16x8 base, four sequential task updates, 64 examples per task, 300 AdamW updates, and development seeds 19901/19902. Both .003 and .01 were run; the preregistered final all-method/all-condition mean MSE selected LR .003 (.16014 vs .16533). Fresh seeds 19911–19913 were never opened because the development byte gate failed.

Controls: hard tie, generic rank-2 coefficients on the same shared plane, private rank-1/2 LoRA, shared rank-1 hypernetwork, and independent full maps. All inference records charge shared state, task IDs, metadata and every tensor. Resume checkpoints include all optimizer moments and step counters.

## D — result

**FAIL at development.** The registered requirement was total inference payload <=0.90x rank-1 LoRA on both development worlds before fresh access. Candidate payload was 1,103B versus 1,196B (0.922x) in both worlds, so fresh stayed sealed. Its incremental state was smaller (80B/skill vs 143B), but fixed shared base and plane bytes prevented the registered total-payload improvement.

Selected-LR aligned means after four tasks:

| Method | Mean seen-task MSE | Total inference bytes | Incremental bytes/skill | Resume bytes/skill | Active compute proxy | Wall time | Inference examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mirror gradient coordinate | 2.53e-4 | 1,103 B | 80 B | 2,119 B | 11.98M | 0.494s | 3.02M |
| Generic shared-plane `P @ C_t` | 2.53e-4 | 1,146 B | 90 B | 1,267 B | 13.52M | 0.362s | 4.51M |
| Rank-1 LoRA | 2.64e-5 | 1,196 B | 143 B | 2,342 B | 11.67M | 0.406s | 6.78M |
| Rank-2 LoRA | 4.86e-7 | 1,580 B | 239 B | 2,678 B | 13.52M | 0.336s | 6.73M |
| Shared rank-1 hypernetwork | 2.97e-2 | 1,744 B | 279 B | 1,541 B | 26.42M | 0.641s | 2.19M |
| Independent full maps | 3.18e-7 | 2,986 B | 589 B | 3,670 B | 9.83M | 0.328s | 15.52M |

Mirror/LoRA rank-1 quality ratios were 18.10 and 0.656 across the two development worlds, so quality was also unstable. The generic plane control had nearly identical mean quality and smaller resume state, 1,267B/skill versus 2,119B/skill. This is the strongest counter-evidence to a Mirror-specific claim. On unrelated task maps, shared methods remained poor; independent full state was much more accurate at higher storage.

## Fact / interpretation / hypothesis

**Facts.** The total inference-byte gate failed in both development worlds. Mirror used fewer incremental inference bytes than rank-1 LoRA but not enough to meet the total-payload limit. Generic shared-plane coefficients matched the candidate's mean aligned quality while using fewer resume bytes per skill. No fresh result was accessed.

**Interpretation.** In this small task bank, record/optimizer metadata for separate angle and vector parameters erased much of their scalar-count advantage. Ordinary coefficients in the same shared plane were the stronger control for optimizer-resume storage. The task-specific angle did not reliably match LoRA quality across seeds.

**Hypothesis.** Packing the angle and vector into one optimizer parameter or changing to a better-conditioned update basis might reduce checkpoint overhead, but that is a new protocol and cannot rescue this frozen development gate. A later test needs a fresh experiment/amendment and a byte-near shared-plane control.

## C — strongest counter-hypothesis

The candidate's separate angle/vector parameter groups caused framework optimizer metadata overhead; a packed parameter representation could lower actual resume bytes. This possibility does not change the observed inference-byte failure or generic-control result.

## U — unresolved

- Fresh behavior is NOT ESTABLISHED; fresh seeds remain sealed by the development gate.
- The shared basis discovery cost from natural gradients was excluded; basis bytes and QR setup were reported.
- No language-model or natural continual-learning task, no near-convergence capacity frontier.
