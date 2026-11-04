# MA-658 — Fast-write Mirror state, slow consolidation

Test whether temporary task-specific views can be consolidated into a shared functional basis while preserving old functions across consolidation boundaries. This screen uses synthetic linear tasks and compares private fast rank-2 adapters, a learned rank-4 Mirror code bank, the identical native low-rank/PCA bank, and independent full matrices.

At each eight-task boundary, the shared basis is recomputed from all observed task operators and all task codes are remapped. Every task is then replayed on held-out inputs. Full payload bytes include basis, codes, task IDs, and schema metadata. The native method performs the exact same matrix operations as Mirror so attribution is directly testable.

This is a synthetic consolidation mechanism screen, not a ReLoRA or Titans reproduction. See `PROTOCOL.json` for frozen conditions and decision gates.
