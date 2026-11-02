# MA-539 status

**FAIL — exact synthetic mechanism screen; fresh worlds completed.**

- H: a single shared function code can drive all P=4 ordered slots and reduce repeated address storage; a function-vector-specific advantage over generic shared latent was also tested.
- T: two development worlds (60101–60102), then three frozen fresh worlds (60111–60113), 64 support pairs and 256 query packets/world. Exact finite-state shared rule bank/executor; no optimizer updates.
- D: shared FV address reached 100% exact packet accuracy in all five worlds. Address payload was 336 B versus 1,104 B for four independently repeated slot address tables (30.4%). However FV, PTP-style shared latent, generic latent and charged rule-ID oracle serialized to the identical address tensor/hash and had identical outputs. Therefore the FV-specific advantage FAILS; shared-code storage factorization is a generic-address result, not Mirror/FV-specific.
- C: the task hands each rule a separate support set over a known shared rule bank. This makes exact categorical identification easy and the FV implementation collapses to the ordinary rule ID.
- U: no learned/causal LLM function vectors, generalization to unseen functions, stochastic within-packet uncertainty, joint NLL, neural quality, or method-specific throughput tested. Exact common finite-state executor calibration was 0.152–0.175 s for 1,000 passes over 256 packets; it does not separate methods.

**Protocol deviation:** amendment-1 was fixed before any measurement and the initial protocol had been committed, but the amended protocol/code were not committed before the first development run. The amendment, run, and exact provenance are retained; treat this as a procedural deviation. Fresh worlds were opened only after the amended development byte/accuracy gate passed.

See `README.md`, `PROTOCOL.json`, `RESULTS_CORE.csv` and `runs/` for facts and hashes. The result is a mechanism boundary, not a capacity or language-model claim.

**Storage accounting amendment:** canonical inference JSON metadata is now included. The dev/fresh quality outcomes did not change; complete payload counts are 1,675 B shared vs 2,452 B independent.
