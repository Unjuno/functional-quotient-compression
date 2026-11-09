# MA-695 implementation amendment

The first implementation run divided latent-space attention logits by `sqrt(R)` after absorbing the K up-projection into the query. The explicit materialized baseline divides by `sqrt(D)`. Since the latent dot product is algebraically the same score as the original head-space dot product, its scale must remain `sqrt(D)`. The initial source and RESULTS_CORE.csv are preserved under `source/attempts/initial_bad_scaling/` and are invalid for scientific conclusions.

Correction changes only this implementation error. The frozen dimensions, targets, seeds, controls, thresholds, and analysis grid remain unchanged. The corrected source replays the same already-locked development/audit seeds; no hyperparameter or selection decision was made from the invalid run. This amendment is recorded after the initial measurement and before corrected result interpretation.
