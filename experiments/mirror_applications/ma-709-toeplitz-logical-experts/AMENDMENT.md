# MA-709 amendment — model-level serialization accounting

The first implementation serialized one NPZ per logical task and used those per-task sizes as the authoritative payload. That repeated container/metadata overhead and did not represent one deployed model payload. The entire initial per-task RESULTS_CORE.csv and serialized payload set is retained under `source/attempts/per_task_container_accounting/` and is invalid for storage conclusions.

The protocol now serializes one complete inference payload per method, seed, and private-variation level, including all maps and shared state in one archive. The mathematical models, seeds, task draws, metrics, and gates are unchanged. Because the initial audit outputs were visible before this accounting correction, corrected same-seed runs are labeled as amended audit replays; no tuning decision uses their results. This amendment only corrects the authoritative byte unit.
