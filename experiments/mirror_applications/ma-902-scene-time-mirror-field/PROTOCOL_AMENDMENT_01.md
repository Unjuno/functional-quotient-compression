# Protocol amendment 01 — stabilize factor optimization and replace fresh seeds

The unregularized frozen run exposed a scale/gauge optimization instability: development heldout nRMSE was 54.35 for seed 90201 and .0258 for 90207. The originally designated fresh seeds 90211/90213/90219 were also opened before diagnosing the failure. Their results are preserved under `source/results/pre_amendment/`; they are exploratory only and are excluded from tuning and confirmatory claims.

Development-only sweep on seeds 90201/90207 used Adam weight decay in {1e-4, 1e-3, 1e-2}; heldout nRMSE was respectively .02552/.02862, .08992/.10531, and 1.00/1.00. Select 1e-4 based only on development seeds.

Freeze `weight_decay=1e-4`, retain the other protocol settings, and use new fresh seeds 90221/90223/90229. This amendment is recorded before generating/evaluating those fresh seeds.
