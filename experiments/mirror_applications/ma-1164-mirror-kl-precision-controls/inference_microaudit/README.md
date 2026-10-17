# MA-1164 follow-up — packed inference state and matrix-free Mirror execution

**Status:** POSTHOC DIAGNOSTIC; no modification to MA-1164's historical M0/FAIL result.

**Mirror insertion:** a shared physical coefficient vector `u` and deterministic, distinct Mirror codes are inserted at the virtual-weight generation / implicit-operator application interface, replacing physically duplicated per-head/per-depth matrices while retaining the executed computation in an explicit reference implementation.

Two **separate** mechanisms are tested:
1. **Existing learned-weight decoder's serialization only:** replace costly int16 position lists with a bitset, store only FP16 physical coefficients; the existing `source/core.py` still generates all 1210 virtual weights per inference. Compare against equally tightly packed FP16 native MLP.
2. **New implicit computation family:** use a sign-addressed Walsh-Hadamard orthogonal Mirror and one shared coefficient vector. Directly apply its virtual matrix without instantiating it, and verify exact equivalence to a fully materialized dense reference. Test an additional Mirror–nonlinearity–inverse-Mirror sandwich and its gradient.

Do **not** conflate 1 with 2: the implicit family is not the old trained image classifier, and has not been fitted to its learned weights. It is a computational feasibility demonstration only.

## Run

From this folder with `torch`, `numpy`, `pytest` and project MA-1164 `source/core.py` available:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python source/online_operator.py
```

Optional already-opened old-checkpoint replay (never counted as fresh): `python source/legacy_teacher_audit.py --checkpoint-directory /path/to/ma1163/results`

Results are deterministic except wall-clock jitter and runtime backend differences. Raw binary `.bin` and `.npy` are generated on execution and intentionally not committed. Both record programs require the corresponding versioned executable runtime; shared code bytes are not silently claimed as zero total deployment cost.

See `PROTOCOL_DIAGNOSTIC.json`, `REPORT_JA.md`, and `results/`.
