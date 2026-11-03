# Amendment 2 — restore unpacked matrix width

The amended 57501 attempt stopped during reconstruction before writing any metrics: packed int4 tensors have width 256 bytes per row, while unpacked values have width 512. The decoder returned the packed shape. A test was added for this contract; that test first exposed that the source edit had not changed the decoder, then passed after the source was corrected. Both failed attempts emitted no metrics and are excluded. Protocol and selection semantics are unchanged.

Source SHA-256 before correction: `912db13b1f28e8cbe9830498bf7b505958e73b5b92b8c10e673da54f7eb667e1`
Corrected source SHA-256: `bef53faea2993f6cdbfa75e1a12af2474f2f44b554cc0644630f2e6968824af1`
Protocol SHA-256 remains unchanged: `0754daffab4540278d8e96afb4d9d774c4584115a46de94024db0e80cceb5c94`.
