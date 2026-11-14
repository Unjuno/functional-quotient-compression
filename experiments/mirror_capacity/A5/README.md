# A5: external-teacher Mirror capacity test

**Status:** Completed 2026-11-15. Negative result for the tested shear Mirror4 at fixed whole-weight-file bytes. No general Mirror-native capacity multiplier or practical 64× compression claimed.

- [Pre-run protocol](PROTOCOL.md) — explicit H/T/D/C/U and training/selection boundaries.
- [Japanese research report](REPORT_JA.md) — model family, proof of weight-generation equivalence, variables, uncertainty, all major medians and failure modes.
- [Model, teachers, exact serializer](a5.py), [training](train.py), [held-out audit](audit.py), [tests](test_a5.py).
- [`results/summary.json`](results/summary.json), [`results/contrasts.csv`](results/contrasts.csv), [`results/world_means.csv`](results/world_means.csv), [`results/quality_frontier.csv`](results/quality_frontier.csv) — small, inspectable Git records.
- Large exact-weight evidence archive and full 288 checkpoint files are kept outside ordinary Git history; `EVIDENCE.json` records their SHA-256 and frozen-aggregate provenance. The artifact can be regenerated with the pinned commands in the report.

**Main finding:** Mirror4 lost to the validation-selected non-Mirror control on IID and OOD across all 6 worlds in each teacher/byte-cap condition (24 IID world-level comparisons). Strongest controls were an output basis for compositional teachers and typically FiLM for independent teachers.

**Evidence boundary:** 288 CPU synthetic models, 16 *known* task IDs, 800 fixed steps, and two budget points. These are not a proof of a universal impossibility theorem.
