# MA-502 — Many task codes over a shared LoReFT subspace

Status: SCREENING; protocol frozen before development  
Branch: `research/ma-502-loreft-many-task-codes-20261008`  
Prior art: PA96 (ReFT / LoReFT)

## H — Hypothesis

For exactly aligned rank-four hidden-state interventions, one shared LoReFT-style basis plus one four-value code per task will preserve all tested functions as task count grows from 1 to 256 and use at most 0.30x independent rank-four bytes by eight tasks.

## Mirror insertion

**Mirror insertion:** this experiment adds a four-value task code `m_t` to one shared low-rank representation intervention so that many distinct task functions can be called without storing a separate pair of rank-four factors for each task.

This isolates logical-task count and storage amortization; MA-501 already tests private-variation quality loss. Compare exact native shared LoReFT coefficient storage, independent rank-four factors and independent full matrices. Only directly evaluated task IDs count as functions. The exact protocol/source hashes are in `freeze.json`.
