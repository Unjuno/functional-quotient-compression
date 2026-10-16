# DMFFN-GROK-001 — Does gate×up Mirror SwiGLU exhibit grokking?

Executed **30 locked fresh CPU training fits** on four modular arithmetic functions, plus **6 post-primary shared-only control fits**. Check [Japanese report](REPORT_JA.md), [frozen primary protocol](PROTOCOL.json), [SHA lock](PRE_FRESH_LOCK.json), [primary summary](results/fresh/ANALYSIS.json), [learning curves](results/fresh/LEARNING_CURVES.csv), and [audit](results/fresh/AUDIT.json).

**Mirror insertion:** The two input projections of one shared 3-matrix SwiGLU are separately rotated with two low-description gate View codes and two low-description up View codes, creating four compositional tasks without four physical MLP blocks. Native equal-count gain and per-pair code controls isolate the Mirror coordinates.

**Observed:** A 3-split × 2-initialization 8,000-update CPU comparison gave 60.0% test accuracy for Mirror versus 35.2% gain, 35.7% pair-specific Mirror, and 82.3% independent 4 MLPs. Mirror delayed generalization criteria passed 5/6 seeds, but **strict 90%-test strong grokking passed 0/6**; 0/30 across all five primary methods. Standardized inference state payloads: Mirror/gain/pair=43,597 B, independent=163,861 B. Equal accuracy compression and Mirror-unique grokking **not established**.

**Reproduce in a clean output directory:**

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests
python batch.py --out results/fresh-reproduction --workers 3
python audit.py --path results/fresh-reproduction
python analyze.py --path results/fresh-reproduction
```

Do not overwrite archived `results/fresh`. Primary source hash lock `PRE_FRESH_LOCK.json` must match. `retrain.py` checks two primary fits including all saved checkpoints and full checkpoint payload bytes; `audit_tied.py` checks the separately frozen posthoc tied baseline. All software except Torch/Numpy/SciPy/pytest is Python standard library.

`main`, old DMFFN reports, canonical MA registry and historical failed gates remain unchanged. Three data splits are drawn from the same modular universe and are not independent real-world datasets. No GPU, language-model, or multi-token decoder speed claims.
