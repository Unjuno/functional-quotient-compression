# GIC002 — Mirror complexity, routing initialization and function-preserving growth

Completed container experiment, 2026-10-16. No main merge, canonical MA status change, historical rerun or general LLM claim.

Mirror insertion: two soft-routed expert functions share an MLP and use 1/4/8 adjacent-plane Givens conjugations around GELU. Matched direct paired gains and independent experts are controls. Mirror count stays2.

Read [Japanese summary](SUMMARY_JA.md), [mathematical contract](MODEL.md), [primary protocol](PROTOCOL.json), [widening protocol](WIDEN_PROTOCOL.json), and [master verification](MASTER_VERIFICATION.json).

Primary:2240 models across5 worlds x8 inits x4 strata x14 methods; calibration224. More directions pass quality in all4 strata but absolute-variance+quality gate in only1. Forced uniform warmup fails that gate in all4; the secondary equal-initialization-only contrast improves aligned512 and is reported separately. Native gain wins on unrelated teachers. Extra directions do not prove earlier starvation caused old failures.

Widening:320 continuations from80 common parents on5 new worlds. Zero-angle extension preserves predictions exactly and passes quality versus narrow continuation in both families,5/5 each. Post-hoc same-final-capacity references (160 models) show growth is not universally better or more stable than starting wide. Relative variability remains; all alternative explanations are retained.

19 tests;5808 finite payloads; independent NumPy/SciPy replay;40 exact retrainings. Fixed source locks are retained. CPU FP32, Intel Xeon Platinum8370C,1 thread/worker, two concurrent workers, clock not locked. All2 experts evaluated; not sparse or equal-time training.

Install requirements and CPU torch2.10.0, then run `python -m pytest -q`. Reproduce into a NEW path with `python reproduce.py --out /tmp/gic002-reproduction`. Source/scripts regenerate raw records and payloads; complete original evidence is in the conversation artifact GIC002_20261016_FULL_EVIDENCE.zip. Never overwrite supplied original evidence.
