# Amendment 1 — repair pre-result selection implementation

The initial MA-575 development attempt completed the frozen error-table calculation but stopped before writing `metrics.json` because NumPy in the container does not accept tuple axes for `argmin`; inspection also found the independent and layer-shared controls selected codes using the factor-pair error table but serialized a single candidate code. No completed registered result was emitted.

Fixes: reshape the factor-pair table only where needed; calculate each single-candidate code's own calibration/audit error; select independent and layer-shared control IDs from that direct-code error table. The hypothesis, split, candidate family, controls, two-sweep factor fit, gates, seeds, and outputs are unchanged. The failed attempt is excluded. This amendment is recorded before rerunning either registered development seed.

Original frozen source SHA-256: `a1bcc62a838803bdbf54da77f2580b4cec3e2827a8f6138516782ea30537e2aa`
Amended source SHA-256: `912db13b1f28e8cbe9830498bf7b505958e73b5b92b8c10e673da54f7eb667e1`
Protocol SHA-256 remains unchanged: `0754daffab4540278d8e96afb4d9d774c4584115a46de94024db0e80cceb5c94`.
