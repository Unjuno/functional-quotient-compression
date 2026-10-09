# MA-449 — Concept-coordinate meta-learning

## H — Hypothesis

Support-inferred factorized Mirror coordinates will extrapolate to a held-out joint concept combination within 1.10x additive task-vector quality and use fewer serialized bytes than LEO.

## T — Test

Two-factor synthetic linear tasks. Fresh worlds 44910–44912, three seeds, 20 held-out joint tasks/world-seed. Each world provides single-factor interventions to identify the shared decoder. Compare direct vector composition, gradient-adapted LEO code, least-squares Mirror code, and full-space least squares. A1 scopes this as a mechanism test rather than a full neural MAML/LEO reproduction.

## D — FAIL

Mirror NRMSE 2.91e-07 matched direct task-vector 2.94e-07; LEO was 0.352. Mirror actual serialized payload was 107.45B/task, exactly the same as task-vector and LEO. It passes factor identifiability but fails the registered Mirror-specific byte gate.

## C — Strongest counter-hypothesis

The factors are additive, so ordinary task-vector composition already provides the useful function.

## U — Unknown

No natural concept or nonlinear model evidence.
