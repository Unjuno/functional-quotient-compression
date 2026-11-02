# MA-571 — DEQ equilibrium Mirror coordinate

PA124 makes solver iterations and equilibrium cost explicit. This CPU synthetic screen uses a contractive fixed-point map with two role-conditioned equilibrium functions; it records residual, iterations, serialized state and compute. No trained DEQ or natural task is claimed.


## Fresh result

Mirror and direct-angle reach mean NRMSE around 1.1e-8 in three worlds, with about 2,834 B vs 2,987 B independent role maps (94.9%). The direct code exactly aliases Mirror; the registered <=50% storage gate also fails. **Disposition: FAIL for Mirror-specific benefit.**
