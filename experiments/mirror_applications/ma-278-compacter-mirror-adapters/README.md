# MA-278 — Compacter Mirror hypercomplex adapters

Status: **FAIL** (screening; fresh worlds sealed)  
Base commit: `f7f76de`  
Prior art: PA23, Compacter (shared slow Kronecker weights with task-specific low-rank/rank-one fast factors).

## Hypothesis

A per-task compact Givens coordinate over shared fast factors can express useful adapter specialization around shared Compacter slow weights at lower payload than task-local fast factors, and beat scalar coefficient modulation.

> **Mirror insertion:** this experiment adds task coordinate `m` to paired shared rank-one fast factors in a Compacter-style Kronecker adapter so that six logical adapters can differ without storing separate fast factors per task.

## Method

Synthetic 16-to-16 linear regression with six task adapters, rank-2 Kronecker slow basis, 8192 training inputs and 2048 held-out inputs per task. Development worlds 27800/27801; 500 Adam updates at LR .003/.01. No teacher values are copied into student initializations. Controls: tied, native Compacter, shared-factor scalar coefficients, Mirror Givens coordinates, independent full adapter upper reference. Fresh worlds 27802–27804 were preregistered and remain sealed because the development access gate did not pass.

Protocol and exact conditions: [PROTOCOL.json](PROTOCOL.json). Runner: [source/run.py](source/run.py). Results: [RESULTS_CORE.csv](RESULTS_CORE.csv).

## Decision

See [STATUS.md](STATUS.md) for H/T/D/C/U and separated Fact/Interpretation/Hypothesis. Mirror missed the access gate in both development worlds at both learning rates. It matched scalar modulation bytes but had substantially higher error than both scalar and native Compacter. The evidence supports a FAIL for this unaligned synthetic screen; it does not test aligned feasibility or prove impossibility for Compacter generally.


## Cross-worker adjudication in baseline `28194ea`

A recorded random draw selected MA-278 from `[MA-274, MA-278, MA-282, MA-286, MA-296, MA-299]` (Python `secrets.randbelow(6)`, index 1). A live completed branch existed at `research/ma-278-compacter-mirror-adapters-20261008`. Its frozen protocol, two development worlds, both learning rates, code and reported FAIL were reviewed. This report is selectively integrated from result commit `c159672` and verification follow-up `1d93af7`; the branch as a whole is not merged because it deletes current-baseline research docs. Fresh seeds remain sealed under the originating development access gate. Serialization roundtrip and independent metric replay were not run in the source branch; the result is retained as a screening FAIL at that evidence strength.

Local reverification in this container could not import `torch` (`ModuleNotFoundError`). The origin branch records 3 pytest tests passed. No fresh training was rerun here; this is a reviewed evidence import, not an independent replication.
