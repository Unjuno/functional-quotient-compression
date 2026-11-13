# C4 task-conditional Mirror diagnostic (MA-951 / MA-952)

## Scope

Two synthetic mechanism screens shared a C4-oriented image task with two exact orbit-related orientation labels and one non-orbit object label. Exact transformed copies are not counted as independent functions. These tests do not establish a general G-CNN or steerable-CNN limitation.

## Facts

- MA-951: fixed phase failed its frozen task0/task1 retention gate on 2/3 seeds; serialized payload was 6,432 B vs 6,425 B for independent heads. Ordinary task gate also missed the orbit tasks. Adding a private task-2 residual restored quality but raised payload to 6,851 B.
- MA-952: e2cnn C4 steerable task×irrep code and per-channel ordinary gate both averaged 71.7% on orbit tasks; independent readouts achieved 100%. Mirror payload was 22,917 B vs 22,781 B independent. Private residual did not restore orbit-task accuracy.
- Both screens had near-zero representation equivariance error and exact serialized replay; this verifies implementation mechanics, not useful task multiplicity.

## Interpretation

Across these two small screens, compressed shared-readout task modulation did not reproduce independently parameterized task readouts, while ordinary gating was similarly limited. Private state did not improve the group-related task gap. The repeated outcome is consistent with a shared-readout expressivity/optimization bottleneck and no demonstrated byte frontier improvement.

## Decision

Pause the direct task-specific soft-symmetry-breaking screen family (MA-953..960) for structural redesign. Re-entry should add a preregistered mechanism capable of changing the task readout beyond scalar/diagonal modulation and compare against a matched ordinary low-rank or private-head control. Continue with a different family (MA-961 spiking thresholds).

## Hypothesis / uncertainty

The shared head may be too constrained, or 500 updates may be insufficient. A richer nonlinear task decoder or task-conditioned feature extractor could work, but that is a new mechanism and must be tested on fresh data. No broad capacity conclusion follows from these synthetic screens.
