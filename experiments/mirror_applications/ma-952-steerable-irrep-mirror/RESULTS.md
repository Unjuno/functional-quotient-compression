# MA-952 results — steerable CNN irrep-channel Mirror specialization

## H — Hypothesis

A compact task×irrep code over a shared C4-steerable CNN can specialize group-related tasks with fewer serialized bytes than independent task readouts. An ordinary per-channel task gate is the simple control; the non-orbit task may require private state.

## T — Test

**Fact.** CPU synthetic screen using official e2cnn 0.2.3, upstream commit `022d6ca4a78ab666f3c149d1849e324ed1d40238`. One shared C4 steerable convolutional trunk outputs real irrep blocks. Three fresh seeds (95221, 95223, 95227), 512 train examples and 256 test examples per seed, 500 Adam updates, batch 64, lr .003. Training saw orientations 0–2; held-out set includes orientation 3. Two task labels are exact C4 orbit copies; the third is a separate object class. Compared independent task readouts, shared readout plus learned task×irrep scalars, the same shared readout plus a per-channel learned task gate, and irrep code plus task-2 private residual. Amendment 01 aborted after one native-head row because in-place code assembly broke autograd; that exposed seed was excluded and amendment 02 used three new seeds.

## Results

**Fact.** Means over amendment 02 seeds:

| Condition | Task 0 accuracy | Task 1 accuracy | Task 2 accuracy | Payload bytes | Train seconds | Inference examples/s | Dense MAC proxy/example |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent readouts | 1.000 | 1.000 | 0.850 | 22,781 | 17.19 | 4,274 | 626,724 |
| Task×irrep Mirror code | 0.717 | 0.717 | 1.000 | 22,917 | 15.09 | 3,569 | 626,724 |
| Mirror + task-2 private readout | 0.717 | 0.717 | 0.878 | 23,235 | 16.07 | 3,824 | 626,736 |
| Ordinary per-channel task gate | 0.717 | 0.717 | 0.891 | 23,071 | 22.90 | 4,285 | 626,760 |

**Fact.** The Mirror code and ordinary gate had identical mean orbit-task accuracy and both missed the independent-head result on every fresh seed. The Mirror full payload was 136 B (0.60%) larger than independent readouts. Private residual did not restore orbit-task quality. Equivariance maximum absolute error was at most 1.32e-6; serialized reload replay error was 0.0. The run logged upstream e2cnn's deprecated uint8-indexing warning. Dense MACs are a simple proxy, not the steerable basis-expansion operation count. CPU wall-clock varied substantially and does not show a reliable speed advantage.

**Interpretation.** The learned irrep code provided no useful task separation beyond the ordinary gate in this screen. The independent task readouts were both smaller in actual serialized bytes and better on the orbit tasks. This points to a shared-readout bottleneck in this setup; it does not establish a general limitation of steerable CNNs or task-conditioned equivariant models.

## D — Decision

**FAIL (scoped synthetic screen).** The preregistered quality and actual-byte gates failed. No Mirror-specific, compression, or capacity advantage is established.

## C — Strongest counter-hypothesis

The representation may be adequate but optimization of this 500-update head-only comparison may have landed in a symmetric solution: both learned code and ordinary gate retained the same task-0/task-1 behavior. A different code initialization, richer nonlinear head, or longer training could change the outcome, but would need a new preregistered amendment and fresh data.

## U — Unconfirmed

No natural-image dataset, trained benchmark checkpoint, near-convergence frontier, richer irreps, task-generalization study, or deployment-kernel benchmark was tested. Task 0/1 are transformed copies and are not independent functions. Results do not generalize beyond this synthetic C4 setup.

## Implementation incident

Amendment 01 failed before a complete matched comparison due to an in-place autograd mutation. Its first native-head row is not part of the reported result. Amendment 02 was fixed and run on three new seeds; both histories are retained in `PROTOCOL.json`.
