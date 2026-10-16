# GIC001 — Gradient conflict, representation limits and selective spline capacity

**COMPLETED, 2026-10-16. Container CPU experiment; not a rerun of AMA001, a deep KAN, or an LLM result.**

See [the Japanese report and proofs](REPORT_JA.md), [primary results](RESULTS_CORE.csv), [follow-up results](FOLLOWUP_RESULTS_CORE.csv), [verification](VERIFICATION.json), and [follow-up verification](FOLLOWUP_VERIFICATION.json).

## Mirror insertion and native control

A bank of eight nonlinear univariate functions shares one 17-knot linear-spline parent and two learned curve directions, with two coordinates per task. Two selected tasks may receive paid private curves. Knot locations are fixed. Ordinary low-rank curve factorization is an exact native comparator: it matches Mirror initialization, training, functions and serialized bytes. This experiment does not claim a new algorithm from renaming it.

## Executed comparisons

Calibration: 192 runs. Primary fresh: five independently generated worlds x three initializations x four task families x twelve methods = 720 runs. Each run uses 600 full-batch Adam updates, 128 training inputs and all eight tasks. Validation/test inputs are disjoint. Follow-up: 285 runs on five NEW worlds, plus 30 common 150-step prefixes. Diagnostics: 60 double-zero-factor/initialization runs. Full deterministic retraining: 48 primary models.

Primary source and hyperparameters were locked before fresh evaluation. The allocation follow-up was designed after primary observations, locked separately, and tested on new worlds. It is not retroactive confirmation. No AMA001 seeds 5201-5205 were used.

## Results and boundaries

- Negative task-gradient inner products persisted when ordinary rank-two learning had already attained its exact empirical class optimum. The largest optimization gap was 3.60e-10 on partial sharing and 1.73e-9 on unrelated curves. Gradient conflict alone is not evidence of optimizer failure.
- PCGrad did not pass the fixed-capacity quality gate; mean test error worsened in all five partial and all five unrelated worlds.
- Validation-residual private selection beat random selection at identical bytes in all five partial worlds, but identified both true exceptions in only 12/15 runs.
- On new worlds, gradient-conflict allocation beat the residual selector under the joint worst-task/mean-loss gate in only 1/5 partial and 2/5 unrelated worlds: FAIL for incremental value.
- Correct one-sided initialization opened a blocked factor path without adding parameters. This does not show old repository implementations had that bug.
- Native low-rank and Mirror payloads matched exactly in 60 pairs: FAIL for Mirror-specific value. Rank-two storage is 801 B versus 1254 B independent; two private curves raise it to 1193 B, only 4.86% smaller.

## Reproduction and evidence

Run in a new copy of this directory: `bash reproduce.sh`. It regenerates result files and locks; preserve the original evidence copy. Install `requirements.txt` first when needed. Python 3.13.5, NumPy 2.3.5, OpenBLAS 0.3.30 ILP64, FP64, single thread; AMD EPYC 9V74, clock snapshot 2596.134 MHz, clock not locked. PyTorch 2.10.0+cpu was used for independent gradient/Adam tests. No GPU or quantization. Equal updates are not equal compute.

12 tests passed. All 1197 saved inference payloads were checked by size, SHA256 and an independent interpolation path, with maximum task-metric difference 8.88e-16. Forty-eight complete retrainings matched state and payload bytes. This is same-session verification, not external replication or the historical repository-wide test suite.

Source, protocols, frozen locks, reports, summaries and verification are in GitHub. Complete per-run CSVs, histories, checkpoints and logs are in the conversation's `GIC001_20261016_FULL_EVIDENCE.zip`; they are reproducible from this source. Archive hashes are recorded in `ARCHIVES.json`. The full evidence archive also contains `RESULT_TABLES.zip`: per-world CSV snapshots, descriptive paired intervals, and initialization/runtime diagnostics. Only source, reports, summary CSVs, locks and verification are hosted on the research branch; raw per-run records and checkpoints are in the archive.

**No main merge or canonical MA-status change.** Parent evidence audit: `1c7047843dd093ca20a04d49992946ea58345fe6`.
