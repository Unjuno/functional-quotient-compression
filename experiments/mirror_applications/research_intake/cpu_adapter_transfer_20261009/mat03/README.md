# MAT03 / MAT03R: one real shared input, five output heads, Mirror and native comparisons

**Status:** Mechanism screened, first attempt and separately preregistered new-seed replication completed. **No Mirror-specific Pareto gain; original parent MA status UNTESTED.**

This fixes the important input mismatch of MAT01/MAT02. Every binary role consumes exactly the same input image. A forward hook verifies the expensive trunk is called once, for all nine models including ordinary native five independent output heads.

- [REPORT.md](REPORT.md): methods, all nine full fresh averages, source preregistration distinction, replay, and negative native comparison.
- [MAT03 initial protocol](../MAT03_FROZEN_PROTOCOL.json), [MAT03R replication protocol](MAT03R_FROZEN_PROTOCOL.json) committed prior to its new fresh trial.
- [source/run_mat03.py](source/run_mat03.py), [source/test_mat03.py](source/test_mat03.py): frozen code, 10 tests.
- [source/run_mat03_replica.py](source/run_mat03_replica.py): new seed set without modifying core model/training implementation.
- [results/](results/): initial development 27, initial fresh 45 and replication fresh 45 raw method-world rows, compact core, verification SHA256 and deterministic six-cell replay.

**Decision:** Structured 4D View produced different useful outputs, but ordinary five-head one-pass model was more accurate and had competitive speed, with just 1.35% more total persisted bytes than Mirror4. A source-frozen independent replication did not confirm the initial narrow code advantage above the locked 0.01 nat/label margin. Native one-pass shared trunk is not a hypothetical counterfactual: it was implemented and timed. All results are from repeated partitions of the same small public digits dataset; not LLM, MoE, natural language, GPU or independent dataset-world replication.

No worker queue, registry or main was changed.
