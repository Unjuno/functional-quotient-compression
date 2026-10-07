# SRM003: reproducible causal discovery and pruning

The registered plan was executed, but the scientific composition/adoption gate failed. Read [the report](../../../docs/phase2/SRM003_CAUSAL_DISCOVERY.md), [PLAN.md](PLAN.md), [FROZEN_PROTOCOL.json](FROZEN_PROTOCOL.json) and [ALL_RESULTS.csv](ALL_RESULTS.csv).

Two causal attention blocks,32 hidden units,4 heads, no ES, no recurrent training. Every learner receives only input tokens, not oracle router labels or private masks. Teacher tables are research data and absent from inference payloads.

## Reproduce from this directory

CPU Python3.13, PyTorch2.10.0+cpu, NumPy2.3.5, pytest9.0.2 were used. Do not compare timings across hardware as if they were equal compute.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q
python source/engine.py --out results/fresh66001 --world 66001 --init 1701 --fresh --config CONFIGS.json
python source/engine.py --out results/fresh66002 --world 66002 --init 1702 --fresh --config CONFIGS.json
python source/engine.py --out results/fresh66003 --world 66003 --init 1703 --fresh --config CONFIGS.json
python source/diagnostics.py --root results/fresh66001 --world 66001 --seed 1701
python source/diagnostics.py --root results/fresh66002 --world 66002 --seed 1702
python source/diagnostics.py --root results/fresh66003 --world 66003 --seed 1703
python source/run_pair_diagnostic.py
python source/replay.py
python source/audit.py
python source/summarize.py
python source/benchmark.py
```

The training stream must be generated with the recorded total length: changing that length changes pair RNG consumption. Main and long-continuation streams deliberately use separate fixed seeds and are saved in full. Do not change total length and assume prefix identity.

The two-call replay is an evaluation counterfactual with additional compute and supplied execution order; no true intermediate labels are fed to it. It is not the one-forward main model. The low-rank comparator updates the base too and is not original frozen-base LoRA.

Full data/checkpoints/optimizer states, development runs, Japanese report, SHA manifest and audit logs are in `SRM003_CAUSAL_DISCOVERY_RESULTS_2026-10-07.zip` on the conversation surface. Repository commands rebuild fresh results without downloading that archive. PyTorch pickle-based research files should only be loaded from trusted provenance. The custom JSON/tensor-byte decoder is a fixture codec, not a hardened untrusted-file parser.
