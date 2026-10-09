# MA-208 — distill specialist teachers into a Mirror view bank

Status: **FAIL at development gate; out-of-gate fresh runs are exploratory only**  
Branch: `research/ma-208-mirror-expert-distill-20261007`  
Base commit: `621dec2ffdf55c3f0503dd75695ea27fdc68544c`  
Protocol freeze: `faad9ec1a1af5c474e7bdff39829931b8e221972`; measurement amendment before fresh access: `d421074e08a8f83e8e24eccf37bfc143ea7aa88f`

## H — falsifiable hypothesis

A shared student MLP with one learned hidden-pair Givens coordinate per specialist can distill related teacher experts at ordinary multi-head KD quality with substantially fewer inference bytes. Independently sampled teacher functions should show where private student weights are needed.

## Prior-art delta

MoE-to-dense work such as *Pruning and Distilling Mixture-of-Experts into Dense Language Models* (arXiv:2605.28207) selects expert knowledge, initializes a dense student, and distills teacher logits. Recent multi-teacher on-policy distillation work includes *Rethinking Self-Distillation for Multi-Teacher Capability Merging* (arXiv:2610.04272) and verifier-gated teacher assignment (arXiv:2609.15404). PA09 uses a shared trunk with multiple future-token heads; PA10 conditions parallel token prediction on sequence randomness. These works establish shared-trunk, multi-head, and teacher-selection controls, but do not test whether related teacher functions can replace private student heads with a compact hidden Mirror coordinate.

## T — protocol and execution

Five task IDs use 12D inputs, 16 hidden units and four output classes. The aligned family reuses one teacher MLP and applies a Givens rotation to the first hidden pair before the shared output projection. The independent family samples each teacher MLP separately. Students received 128 examples per task and 300 updates, using temperature-2 forward KL. Controls were frozen hard tying, sequential single-head KD, per-task output-head KD, rank-2 LoRA, a rank-2 hypernetwork, independent full students, and exact teacher functions. Development seeds were 20801/20802; LR candidates .003/.01. The preregistered aggregate KL selector chose .003 (0.0263439 vs 0.0278295 across all methods, conditions and development seeds).

The protocol required both (a) Mirror mean teacher KL <=1.10x multi-head and <=1e-4 absolute, (b) top-1 agreement >=.99, (c) ECE within .02 of multi-head, and (d) total serialized inference payload <=.50x multi-head in each aligned development world before fresh access.

## D — result

**FAIL at the development gate.** At LR .003, Mirror passed the top-1, ECE, absolute-KL and byte subgates in both aligned development worlds, but failed the registered relative-KL requirement:

| Development seed | Mirror KL | Multi-head KL | KL ratio | top-1 agreement | ECE delta | payload ratio |
|---|---:|---:|---:|---:|---:|---:|
| 20801 | 1.77e-6 | 4.56e-8 | 38.8x | 0.9989 | -0.00086 | 0.476x |
| 20802 | 3.91e-7 | 3.80e-10 | 1,027.7x | 0.9993 | -0.00071 | 0.476x |

Both development worlds therefore failed the quality gate. Under the frozen protocol, fresh seeds should have remained sealed. They were mistakenly run after the absolute-KL, top-1, ECE, and byte checks were read as sufficient. The out-of-gate runs are retained in `RESULTS_CORE.csv` with `split_role=exploratory_fresh_opened_after_development_gate_failure`; they are excluded from the status decision and are not confirmatory evidence.

In those exploratory runs, Mirror absolute KL remained low (1.38e-6, 8.31e-10, 3.29e-8) and top-1 agreement was 0.9984–1.000, but relative KL versus the near-zero multi-head errors was 447.9x, 5.51x, and 80.7x. The byte ratio remained 0.476x. These observations do not repair the registered failure.

Selected-LR exploratory aligned means: Mirror used 1,322 B versus 2,775 B multi-head (0.476x) and 2,226 B rank-2 LoRA (0.594x). It had 4.72e-7 mean KL, 0.99945 top-1 agreement, and 0.00052 lower ECE than multi-head. Its active-compute proxy was 0.852x multi-head, but measured throughput was 0.435x multi-head and 0.660x LoRA. Mean train wall time was 0.935s for Mirror versus 0.656s multi-head and 0.718s LoRA. Resume payload was 9,112 B versus 18,332 B multi-head and 16,476 B LoRA. On independent teachers, exploratory Mirror mean top-1 agreement was 0.402 and KL was 0.0603; independent full students reached 1.0 agreement and zero KL at 7,212 B.

## Fact / interpretation / hypothesis

**Facts.** The registered development quality gate failed in both seeds because KL exceeded 1.10x multi-head, despite passing the absolute-KL limit. The development byte ratio was 0.476x. Fresh seeds were opened after this gate failure; split integrity is false. Exact reruns reproduced all recorded fresh quality, compute-proxy, and payload metrics; wall time and throughput varied as expected.

**Interpretation.** The task-specific hidden angle is a compact code for this deliberately planted Givens orbit. The ordinary multi-head control nearly reproduces the teacher exactly, so the predeclared relative-KL requirement is much stricter than the absolute error bound. The observed storage reduction does not outweigh failure of the registered quality gate. Runtime also regressed, and independent functions needed private weights.

**Hypothesis.** A compact view may be useful for specialist banks known to lie on a low-description orbit, but this run does not establish a quality-matched advantage over ordinary multi-head distillation. Any new relative-error floor, generic shared-basis comparison, or longer-convergence comparison requires a separately registered experiment.

## C — strongest counter-hypothesis

The task view was planted into the teacher family, while rank-2 LoRA can represent the corresponding rank-2 output-projection change and ordinary multi-head KD fit it nearly exactly. The very small multi-head KL makes relative KL sensitive to tiny residuals, but that metric was predeclared and cannot be relaxed after seeing results.

## U — unresolved

- Whether the relative-KL gap closes near convergence, or with a byte-matched generic basis, is untested.
- The exploratory runs cannot establish fresh generalization because development failed the opening gate.
- No language-model, natural expert, capacity-frontier, or optimized-kernel evidence.
