# DMFFN-003 — Natural digit SwiGLU Mirror transfer and compression

**Status: EXECUTED RESEARCH, mixed/negative external-validity evidence.** Read [full Japanese report](REPORT_JA.md), [original plan](EXPERIMENT_PLAN_JA.md), and `results/` for complete metrics. No main/canonical MA-status change.

**Mirror insertion:** In a native SwiGLU binary classifier, insert separate low-description Givens view codes immediately after the shared gate and up input projections. Three gate views and three up views yield nine addressable logical classifiers while reusing the shared gate/up/down weights. This does not imply the ability to approximate nine arbitrary independently learned functions. Native factorized gains, pair-specific Mirror, pair-specific output projections and independent FFNs are explicit controls.

## Three prospective phases and a long-budget probe

- A: pretrain **nine truly independent FFNs** on real sklearn digits conjunction labels, then distill the **six seen pairs** into the shared candidate; three heldout pair outputs are never used for student training. Fresh 3 dataset split worlds x2 seeds x6 methods = **36 fits**, 1,800 updates each. Dev world used only to select a common LR.
- B: after reviewing A, freeze separate protocol and learn six *actual* classification labels directly, leaving three pairs unseen; three new split worlds x2 seeds x6 methods = **36 fits**, 2,400 updates.
- C: after reviewing A/B, freeze separate protocol and train all nine labels to quantify **known-task compression**, not zero-shot; three other split worlds x2 seeds x6 methods = **36 fits**, 2,400 updates.
- D: after reviewing C, freeze another protocol; two new split worlds x2 seeds x3 methods = **12 fits**, 6,000 updates, plus 900 and 2,400 checkpoints. Long training overfit on this small dataset.

All splits use a **single 1,797-image digit corpus**; distinct split worlds are not independent source datasets. Teachers achieve 98.73%–98.92% test average accuracy. Student data leakage into heldout pairs was checked in unit tests.

## Key results

At 9 supervised tasks (C), mirror (4,270 parameters, 20,489 serialized bytes) achieved 0.08966 test BCE; native factor gain (4,270, 20,489 B) 0.09127; native **per-pair output head** (4,454, 20,817 B) **0.03990**; nine separate students (37,737, 153,685 B) **0.03199**. Mirror model payload is 86.7% smaller than independent; charging the additional 2,917-byte train-only preprocessing payload in every method makes the full deployment savings **85.05%** (23,406 B vs 156,602 B). It still **fails the same-quality replacement gate**, and the nearly same-byte native output head is superior.

Unseen-pair test BCE: A mirror 2.489, gain 3.248, but **hard tied baseline 0.684**. B direct label mirror 3.092, gain 3.860, hard tied 0.687. Thus a favorable relative gate-vs-gain number does not establish useful transfer. Full details and paired-world ratios are in `REPORT_JA.md` and `results/ANALYSIS.json`.

Long-budget fresh follow-up average test BCE (900 -> 6,000 updates): mirror 0.0855 -> 0.1278; gain 0.1009 -> 0.1587; independent 0.0380 -> 0.0484. More updates did **not** close the gap; validation chose the earlier checkpoint for all 12 trials.

## Run and verify

CPU Python 3.13, PyTorch 2.10.0+cpu, numpy/scikit-learn, pytest. Every source hash is frozen before its fresh worlds. In an isolated output copy (to avoid overwriting original evidence):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python experiment.py --mode dev
python experiment.py --mode fresh
python direct_labels.py --phase dev
python direct_labels.py --phase fresh
python full_supervision.py --phase dev
python full_supervision.py --phase fresh
python long_full.py
python audit.py
python audit_full.py
python audit_long.py
python retrain_verify.py
python analyze.py
python analyze_full.py
```

Reproduction requires each fixed lock from the archived run; do not recompute the locks from edited sources. **All 120 student inference payloads and 14 train-only preprocessing payloads were audited** by hash, actual serialized bytes and re-evaluated metrics; four deterministic student retrainings and one teacher fit matched tensors/payload SHA. Test suite **18 PASS**. Full `.pt` files and additional raw logs reside in the conversation evidence ZIP; only text source and tables are hosted on GitHub.

**Evidence scope:** tiny natural-image digit classification with hand-constructed compositional labels; not a natural-language Transformer checkpoint, production speculative decoding, GPU throughput, exact independent NumPy reimplementation or general-scale compression result.
