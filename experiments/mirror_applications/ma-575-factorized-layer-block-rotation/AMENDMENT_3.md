# Amendment 3 — keep identity control in the original basis

The 57501 run after Amendment 2 emitted metrics, but the control-flow audit found that the identity int4 baseline still passed through the Hadamard transform. That invalidated its label and metrics. This amendment adds an explicit identity path and a unit check for ordinary groupwise int4 reconstruction. The previous output is excluded from all comparisons; no fresh data were accessed. Protocol and gates remain unchanged.

Source SHA-256 before correction: `bef53faea2993f6cdbfa75e1a12af2474f2f44b554cc0644630f2e6968824af1`
Corrected source SHA-256: `c0c39bd048115ac334b9c3fc2c5f106489c9a906435d8edb1641c1a61151d88e`
Protocol SHA-256 remains unchanged: `0754daffab4540278d8e96afb4d9d774c4584115a46de94024db0e80cceb5c94`.
