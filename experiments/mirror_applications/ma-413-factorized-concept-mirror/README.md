# MA-413 — factorized concept Mirror coordinates

## H — falsifiable hypothesis

Two independently trained attribute-specific Givens coordinates composed on a shared MLP will extrapolate to the held-out Cartesian combination at higher accuracy per byte than additive factorized FiLM or a direct combination-code table that has no training examples for that combination.

## T — protocol

Two binary attributes produce four concept combinations. Train only on 00, 10, and 01; hold out all 11 examples from training and development. The teacher composes two disjoint Givens transforms before a shared MLP. Compare shared, factorized Mirror, factorized FiLM, additive rank-one attribute residuals, a direct per-combination Mirror-code table with zero held-out code, and an oracle independent upper control trained on all combinations. Development worlds select LR; fresh worlds evaluate seen and unseen combinations separately.

This is deliberately aligned compositional feasibility, not natural concept generalization.

