# MA-487 status

- Status: FAIL
- Branch: `research/ma-487-lista-mirror-router-20261009`
- Base: `5071b10c`
- Development LISTA trained; fresh worlds 48710-48712 × seeds 0-2 complete

H: Fixed-depth LISTA approximates OMP codes with lower inference work and practical stored state.

T: Same 64-atom dictionary; OMP top-8 vs LISTA depths 1/2/4/8; serialized router weights charged.

D: FAIL. OMP NRMSE .0269 / 15,329B. LISTA errors .821, .742, .644, .358; each payload 214,613B. It misses quality and storage controls.

C: Dense unrolled matrices plus small development set cause poor generalization and high storage.

U: Better LISTA optimization/structured weights and learned task dictionaries.
