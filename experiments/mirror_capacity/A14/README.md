# A14 / A14b: privileged code-reachability diagnostic

**SYNTHETIC / POST A13 EXPLORATION. Oracle uses test labels.** Do not cite the privileged oracle rows as held-out test performance or evidence of a Mirror-specific compression gain.

## Question and answer

Can the trained two-parameter shear-Mirror code manifold represent the tent function tasks when adaptation is given deliberately excessive label access? On the six previously fixed A13 parent worlds, an additional 960-candidate CEM-ES *trained on the A13 IID+OOD audit labels* improved Mirror but it still passed **0/6** quality worlds, while Shift and FiLM passed **6/6**. A posthoc independent-input challenge preserved this pattern; lack of discovered codes is **not** a proof of global unrepresentability.

## Main frozen experiment

- `PROTOCOL.md`: fixed family/world/seeds/methods, method, bounds and decision gates.
- `a14.py`: read A13 parent's original SHA-locked FP32 tensors and the A13 32-example ES codes; run 30 generations × 32 population = 960 privileged new candidates per task in box ±4. Shared parameters and old codes remain frozen.
- `results/{config.json,train.lock.json,audit.csv,worlds.csv,summary.json}`: complete accounting for 36 frozen oracle models and comparison with the A13 ES codes.
- `posthoc.py`: **separately declared post-result** challenge sample generator, 512 new IID + 512 new OOD inputs, entirely unused in A14 candidate fitting. New independent-input scores under `results/posthoc_challenge*`.
- `validate.py`, `test_a14.py`, `test_validate.py`: frozen source/parent/hash guards, model roundtrip, Torch/NumPy parity, deterministic optimizer replay and template construction. 8 tests passed, 4 independent model reproductions byte-identical.
- `REPORT_JA.md`: Japanese report, formulas, variable table, H/T/D/C/U, uncertainty, false-positive and source boundaries.

## Core numbers (six world means, two seeds per world)

| model | 32-example ES IID/OOD | privileged test-label CEM IID/OOD | privileged in-sample world passes |
|---|---|---|---|
| Mirror | 0.362 / 0.478 | 0.333 / 0.412 | 0/6 |
| Shift | 0.093 / 0.102 | 0.077 / 0.087 | 6/6 |
| FiLM | 0.160 / 0.169 | 0.136 / 0.148 | 6/6 |

**Warning:** Privileged scores deliberately train on the test targets; only original A13 support-only ES scores are held-out in the A13 protocol. The challenge was added after inspecting A14 results. The experiment is not a generalization or rate-quality benchmark.

## Reproduce

With Python3.13, NumPy2.3.5, PyTorch2.10.0+cpu, SciPy1.17.0, pandas2.2.3 and the original A13 evidence archive extracted as a sibling at `../mirror_es_a13_20261115` (local standalone) or `../A13` (inside GitHub repository):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q test_a14.py test_validate.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python a14.py run --out /tmp/a14-new-output --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python a14.py audit --out /tmp/a14-new-output
# For archived results in this release:
python validate.py
python posthoc.py
```

No regenerated evidence may overwrite the locked original `results`. Software/decoder bytes are not included in FP32-weight-file capacity.
