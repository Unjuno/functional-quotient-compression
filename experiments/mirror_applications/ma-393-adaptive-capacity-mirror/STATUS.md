# MA-393 Status

**FAIL — development screen. Fresh seeds remain sealed.**

H: a compact per-rare-token Givens code would improve rare-token quality beyond native adaptive dimension allocation.

T: two synthetic frequency-skewed worlds (39301, 39302), five methods, 256 Adam updates each and 32,768 examples per method. Common tokens received 8x sampling probability. Scores and CPU measurements used serialized FP16 state.

D: FAIL. All full-table rare accuracies were 1.000. Mirror rare accuracy was 1.000 and 0.938; adaptive4 was 1.000 and 0.990. Mirror did not improve adaptive4 by the required 5 points and fell 6.2 points below full-table in seed 39302. Mirror payload 3,964B exceeded adaptive5 at 3,774B (1.050x; gate <=0.90x) and was 0.750x full16 (gate <=0.60x). CPU throughput was 0.858–0.862x the fastest adaptive native control (gate >=0.90x).

C: the synthetic classification task saturated for full and native low-dimensional embeddings, leaving little quality headroom. Mirror added per-token state and trigonometric compute without improving rare-token accuracy over adaptive4.

U: natural frequency distributions, language modeling, much rarer tail tokens, fresh worlds, near-convergence capacity, and fused kernels remain untested.

Facts are in `RESULTS_CORE.csv` and `runs/`; interpretation is that the fixed-budget Mirror mechanism fails its quality/storage/runtime screen; later work could test more severe tail regimes only under a new protocol.
