# MA-951 — task-conditional Mirror over an exact C4 representation

Status: FAIL (scoped synthetic mechanism screen). The synthetic image task has three heads: task0 detects one orientation, task1 detects the C4-rotated orientation (an exact transformed copy), and task2 classifies a shape/color attribute that is outside that group orbit. One shared CNN is applied to four rotated inputs; its orientation-indexed feature tuple is exactly C4-equivariant by construction.

Compare native independent task heads, a shared head with two-bit phase codes, phase plus private residual for task2, and an ordinary learned group-slot gate. Exact transformed task copies are not counted as new capacity. This is mechanism-only, not natural-image evidence. See [PROTOCOL.json](PROTOCOL.json).
