# MA-647 — ReLoRA basis rotation schedule

Status: SCREENING. This tests whether restart-specific scalar Views over one butterfly schedule can generate diverse low-rank update subspaces.

Four rank-4 updates compose a rank-16 target. Compare independent ReLoRA factors, independently stored BOFT angle tables, a shared butterfly schedule with one scalar per phase, an identical native shared-angle BOFT control, and a static rank-4 boundary control. Every basis, angle, scalar, coefficient, base weight and metadata item is charged in actual serialized payload bytes.

See `PROTOCOL.json`. Fresh seeds 64711–64713 remain sealed until the frozen gates pass.
