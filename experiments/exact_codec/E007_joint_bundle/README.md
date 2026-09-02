# E007 — Joint tree / precision bundle reconstruction

Status: **mechanism-preserving reconstruction**.

The original handoff reports an 8-leaf, five-state (`5^8 = 390,625`) exact
experiment in which tree-only and precision-only moves both fail to improve the
incumbent, while a joint tree+precision move improves distortion at the same
serialized budget.

The original experiment generator was not included in the handoff archive.
`tests/test_local_bundle.py` therefore constructs a new explicit witness with
the same logical phenomenon. It is deliberately labeled as a reconstruction,
not as reproduction of the original numerical instance.

The next recovery target is the original state-cost/distortion generator or an
independent reconstruction that reproduces the archived numbers under a fully
documented model.
