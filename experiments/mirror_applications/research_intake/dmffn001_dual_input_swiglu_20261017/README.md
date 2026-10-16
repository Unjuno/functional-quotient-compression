# DMFFN-001 — Dual-Input Mirror SwiGLU

**Complete synthetic CPU experiment executed on 2026-10-17.** See [Japanese full report](REPORT_JA.md).

**Mirror insertion:** one shared SwiGLU's gate and up preactivation channels each get a low-description Givens View code; 3 gate Views × 3 up Views yield 9 *compositional logical functions* from three shared physical FFN matrices. Ordinary scalar gain, pair-specific Mirror and factorized LoRA are strong controls. Nine independent SwiGLUs are the unrestricted quality control. All state and code bytes are paid and serialized as actual inference payloads.

- Calibrated on development world 73001 at LR0.018, frozen source and split hashes.
- Primary fresh **480 models** = 5 worlds ×2 seeds ×4 conditions ×12 methods ×2,400 updates. Full audit: 480/480 payload SHA and independent NumPy fp64 value checks. **12 tests PASS**.
- A separately frozen post-primary LONG follow-up **108 models** = 3 new worlds ×2 seeds ×3 conditions ×6 methods ×6,000 updates. All 108 audited.
- **6 full reproducibility retrainings**, inference payload and all evaluation checkpoints identical; 8 aligned teacher oracles exactly inside the model class.
- At 648 parameters and 5,081B, dual Mirror12 vs native gain12 and pair-specific Mirror4 wins 5/5 aligned worlds, and strongly improves heldout gate×up compositions. Teacher aligned to candidate; unrelated teachers prefer native gain and independent models.
- **Negative claims retained**: independent 9 SwiGLU (5,184 parameters, 22,817B) has better fully supervised accuracy; automatic capacity growth misses 0/5 quality-and-storage worlds; eager CPU inference Mirror12 (0.164 ms) is slower than 9 independent FFNs (0.081 ms) for batch64.

## Evidence navigation

- `PROTOCOL.json`, `PRE_FRESH_LOCK.json`: preregistration and SHA256 source lock
- `LONG_PROTOCOL.json`, `PRE_LONG_LOCK.json`: separately frozen longer-budget follow-up
- `source/model.py`: SwiGLU and actual Givens/Gain/LoRA comparisons
- `source/train.py`, `source/long_run.py`: 480 + 108 CPU fits
- `source/audit.py`, `source/audit_long.py`: independent NumPy fp64 output checks and SHA/serialized bytes
- `source/analyze.py`, `source/analyze_long.py`: world-paired quality/byte analyses
- `source/retrain_audit.py`, `source/class_oracle.py`, `source/benchmark.py`
- `results/FRESH_CORE.csv`, `results/LONG_CORE.csv`, `results/*.json`, `results/CPU_INFERENCE_BENCH.csv`
- `tests/`: twelve passing tests.

Full source and all raw payloads are in separate downloadable archives; Github research branch can host source, protocols, compact summary and verification. Prior MA registry/status and `main` untouched. No full language Transformer experiment is claimed.
