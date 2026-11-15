# MA-1030 — SAM2 rare-object memory private fallback

Status: SCREENING. This experiment tests whether a causal one-object private memory fallback can preserve a rare track while the other object memories use a shared SAM2 feature bank and compact Givens views.

Prior art: PA315 (SAM2 streaming memory) and PA316 (SAM2Long memory paths). MA-1027 showed that plain Givens sharing damages multi-object tracking; MA-1028 showed path-only compression barely changes full-session bytes. MA-1030 asks where private state must return and whether a data-driven fallback can localize that cost.

The fallback is selected solely from the current frame's per-object SAM2 memory-feature residual from the shared mean. It is capped at one private object per frame. The scene generator makes object 99 change appearance during a short occlusion. There is no future-mask oracle. Development seed 103000 selects one predeclared threshold; three audit seeds remain unopened until that setting is committed.

See `PROTOCOL.json` for frozen gates, controls, and accounting. This is a synthetic CPU mechanism test, not natural-video evidence.
