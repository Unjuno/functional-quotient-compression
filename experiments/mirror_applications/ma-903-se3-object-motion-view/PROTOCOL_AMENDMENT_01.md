# Protocol amendment 01 — equalize the native SE(3) control

## Reason

The initial native-control implementation stored a table of all 36 composed poses, while Mirror stored the six primitive action transforms and composed them at decode time. That charged the native control for extra address/materialized state and made the storage comparison unfair. The initial results and payloads are preserved under `source/results/pre_amendment/`; they are exploratory and excluded from confirmatory claims.

## Frozen correction

The native control now stores the exact same canonical geometry, six primitive SE(3) transforms, metadata, and composition decoder as Mirror. This is the cheapest ordinary native SE(3) parameterization and should produce a byte-exact alias. Replace fresh seeds with 90331/90333/90339 before generating the corrected fresh worlds. No model selection or hyperparameter changes are made.
