# MA-403 status

Status: **FAIL** at the development gate.

## H
Token-position-generated eight-angle Mirror codes represent smooth token-specific transformations with lower held-out error and total payload than token-wise FiLM.

## T
Shared frozen 16D feature block; token positions `[0,0.8)` for training and `[0.8,1)` for held-out evaluation; five controls; seeds 40301/40302, 400 updates. Fresh 40311–40313 were not accessed.

## D
FAIL. Mirror NRMSE was 0.000370/0.000570 versus token-FiLM 0.613/0.729. Mirror payload was 0.827x token-FiLM and 0.591–0.593x full affine, both outside the preregistered limits. Throughput was 0.255x and 0.049x token-FiLM, below the 0.80 minimum.

## C
The synthetic target exactly follows a smooth affine-in-position Givens family. Quality gain may be specific to that aligned family; the compact code also sits on top of shared bytes that dominate total payload at this width.

## U
Natural token workloads, larger block amortization, optimized kernels, fresh replication.

FACT: all ten serialized-payload metrics replayed. INTERPRETATION: structured conditional rotation is representable with a small code, but this deployment point fails byte/runtime Pareto gates. HYPOTHESIS: larger shared objects could amortize code bytes, subject to a new protocol.
