# MA-079 — continuous learned depth address

Status: PROMISING
Evidence lane: MECHANISM
Base commit: `a761cc84422c323e78ad8290a2ff6581bc39723a`

## Hypothesis

H: A continuous two-coordinate depth address can use one shared block to recover intermediate logical layers not directly supervised, while using fewer bytes and better held-out-depth quality than tying, static per-step LoRA, and generated diagonal modulation.

## Physical-to-logical claim

- Physical object: one 8×8 linear map, reused at eight depths.
- Mirror coordinate: fixed deterministic address features `[z,z²]`; two learned coefficients map the address to a Givens angle.
- Logical multiplicity: eight depth maps; training targets are exposed only for depth indices 0, 2, 5, and 7.
- Failure mode: independent or irregular layer functions need per-layer private capacity; a generic depth-conditioned gain can explain the task more simply.

## Prior-art delta

PA01/PA06 establish tied blocks and per-step generated modulation. MA-076 used one independently learned Givens angle per logical layer on a fully supervised synthetic teacher. MA-079 asks whether a compact address function can interpolate unsupervised intermediate depths and compares a depth-conditioned non-Mirror gain control.

## Comparisons

Ordinary tying, static rank-1 per-step LoRA, shared polynomial depth-addressed diagonal gain, Mirror polynomial address to Givens angle, and untied full matrices. All methods train for 600 updates at LR 0.01 on the same four supervised depths. Evaluation includes all eight depths and separately reports the four held-out indices. Two development worlds are used before the fresh gate.

## Storage and compute

Actual payload bytes come from `torch.save` of method metadata and all inference state tensors. Fixed address features derive deterministically from layer index; all learned generator coefficients are charged. MAC proxy, examples, updates, wall time, and training throughput are recorded.

## Results

The fixed 600-update protocol trained only depth indices 0, 2, 5, and 7. Fresh aligned worlds 79011–79013 were opened after development passed the held-out-depth screen.

### Fresh aligned teacher

| Method | Median held-out-depth MSE | Payload bytes | Median train wall / world |
|---|---:|---:|---:|
| tied | 0.0690 | 1,961 | 0.671 s |
| static rank-1 LoRA | 0.3557 | 2,849 | 1.144 s |
| generated depth gain | 0.0646 | 2,213 | 1.180 s |
| Mirror address | 3.36e-12 | 2,149 | 5.400 s |
| untied | 2.1887 | 3,753 | 1.197 s |

Mirror's held-out error was near zero in all fresh worlds and its serialized state used 42.7% fewer bytes than the untied same-observation model. It was 2.9% smaller than the generated-gain control, while held-out MSE was orders lower. The Mirror MAC proxy was 25% above the shared projection baseline, while eager CPU training took about 4.5x the generated-gain wall time.

The untied control saw targets only at the four supervised depths, just like other methods; its large held-out error is expected because its four unseen matrices received no gradient. It is not a fully supervised all-depth upper bound, so the held-out result must not be described as beating independent capacity.

### Fresh independent teacher

Mirror held-out MSE ranged 1.42–1.43, and did not consistently beat tying or generated gain. The continuous address cannot infer arbitrary independent layer maps from sparse supervision.

## Decision

**FACT:** In three fresh aligned worlds, a two-coefficient polynomial depth address recovered the held-out intermediate maps with median MSE 3.36e-12 from supervision at only 4/8 depths. Payload was 2,149B vs 3,753B for the identically supervised untied control. Generated gain used 2,213B and had median held-out MSE 0.0646. Fresh independent teacher results did not show reliable Mirror quality benefit. 50 rows replayed with exact payload bytes and max metric delta 4.94e-10; tests passed 2/2.

**INTERPRETATION:** A smooth, correctly specified Mirror depth address can interpolate logical depth functions with sparse layer supervision. This is a favorable structural screen. The byte comparison against untied is valid as serialized-state measurement, but its held-out quality is not a capacity comparison because unseen untied layers are unsupervised.

**HYPOTHESIS:** Continuous depth addresses can reduce supervision and weight storage when layer functions lie on the learned view trajectory; irregular layers will require private parameters.

**BOUNDARY:** Synthetic 8D linear maps and a polynomial Givens trajectory; CPU only; fixed 600 updates. Natural language, nonlinear Transformer blocks, full-depth supervised untied quality, learned depth-count extrapolation, and optimized kernels remain untested.
