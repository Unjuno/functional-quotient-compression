# MA-247 — recursive shared block + Mirror depth modulation

Status: **FAIL (development screen)**  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `f11fb3cf08634d1a79c1b7c7366bad49b8828ff2`

## H — falsifiable hypothesis

A single recurrent nonlinear block with one low-description Mirror coordinate per depth step can recover useful layer-specific composition from fewer physical block weights. It must beat hard tying and simple scalar gating at comparable bytes, and be compared with static per-step LoRA and input-conditioned generated LoRA.

## Prior art and controls

PA06, *Ouroboros: Dynamic Weight Generation for Recursive Transformers via Input-Conditioned LoRA Modulation*, reuses a block and has a controller modulate frozen LoRA bases per step. It reports that gated recurrence is essential and compares against static per-step LoRA. This screen includes shared-block recurrence with trained residual gates in every method, plus untied nonlinear blocks, hard tying, a scalar per-step gate, static rank-1 per-step LoRA on the output projection, input-conditioned generated rank-1 LoRA, and per-step Mirror Givens coordinates.

## T — execution

A synthetic teacher ran four recurrent depth steps on 8D inputs with hidden width 16. The teacher shared one nonlinear block, conjugated by independently sampled per-step Givens angles. Six student methods were trained for 1,200 AdamW updates, batch 64, on development world 24700 (init seed 247000), at both preregistered common learning rates, 0.003 and 0.01. The selected rate was 0.01 by mean development MSE across methods. All payloads count serialized state plus deterministic config metadata. CPU: Python 3.12.14, PyTorch 2.10.0+cpu, one thread.

No fresh/audit world was opened: the development screen failed the preregistered quality comparisons, so the protocol stopped before fresh evaluation. The listed fresh seeds 24701–24703 remain unused.

## D — FAIL at the development screen

| Method (LR 0.01) | Final-state MSE | Serialized bytes |
|---|---:|---:|
| Untied blocks | 1.463e-4 | 6,181 |
| Hard tied | 2.012e-3 | 3,043 |
| Scalar gate | 6.384e-4 | 3,366 |
| Static rank-1 LoRA | 1.076e-3 | 3,938 |
| Input-conditioned generated LoRA | 2.614e-3 | 4,373 |
| Mirror Givens views | 2.664e-3 | 3,297 |

Mirror payload was 46.7% below untied, satisfying the storage ratio alone. Its MSE was 18.2x the untied control, 1.32x hard tying, 4.17x scalar gating, and 2.47x static LoRA. At LR 0.003, Mirror also had the worst MSE (8.351e-3). The Mirror result therefore failed the preregistered quality gate already on development data. Although generated LoRA had slightly lower MSE than Mirror at LR 0.01 (2.614e-3 vs 2.664e-3), it used 1,076 more bytes; neither method was competitive with scalar gating or static LoRA here.

## C — strongest counter-hypothesis

This is an optimization and task-family screen, not a test of representational impossibility. The teacher was deliberately aligned to Givens views, yet the learned recurrent Mirror still failed; fixed update budget, recurrent credit assignment, or initialization may have prevented it from finding that solution. The dev failure is enough to reject this configuration under the registered screen, but does not show that depth views cannot work after a different protocol or for a Transformer language task.

## U — not established

Fresh-world robustness, near-convergence capacity, fixed-byte quality frontier, natural-language quality, optimized-kernel throughput, and whether better initialization/optimization rescues Mirror remain untested. Do not infer added capacity from these fixed-budget results.

## Fact / interpretation / hypothesis

- **Fact:** Across 12 development rows (six methods x two learning rates), Mirror's MSE exceeded hard tying, scalar gating, and static LoRA at the selected rate; all 12 replayed metric values and payload sizes matched the recorded rows (maximum MSE difference 4.51e-13; payload bytes exact).
- **Interpretation:** Per-step Givens Mirror coordinates did not provide useful depth differentiation within this registered 1,200-update screen. The storage reduction came with a substantial quality loss.
- **Hypothesis:** recurrent optimization/credit assignment may be the limiting factor; a separate, preregistered experiment would be needed to test that explanation.

## Reproduction and artifacts

Run tests with `python -m pytest -q experiments/mirror_applications/ma-247-recursive-depth-view/tests`. The development metric replay is recorded in `VERIFICATION_REPLAY.json`; protocol freeze and source hashes are in `FREEZE_MANIFEST.json`. No checkpoint is required for the serialized-payload measurements.
