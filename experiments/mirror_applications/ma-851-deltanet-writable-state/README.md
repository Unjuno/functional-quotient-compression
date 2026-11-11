# MA-851 — DeltaNet low-rank writable state View

H: a rank-r writable matrix state preserves associative recall for rank-r value streams with lower serialized state bytes and update compute than a full DeltaNet matrix.

PA230 identifies DeltaNet's matrix state and delta-rule updates; PA25 establishes fast-weight outer-product controls. Mirror insertion restricts writable W to W=U C, with serialized shared value basis U and transient episode code C. Compare full delta rule, native low-rank state, and Mirror representation.

Synthetic D=32, rank r=2, 16-step orthogonal-key episode; 16 stored-key queries. Development seed 19; frozen fresh seeds 41/67. Record actual NPZ bytes, full and low-rank writable state, support updates, active MACs, query MACs, and reset isolation. No optimizer training.

C: native low-rank state is likely function-identical and cheaper than a distinct Mirror code. U: trained DeltaNet, correlated/nonorthogonal keys, long-context interference, and throughput on accelerator.
