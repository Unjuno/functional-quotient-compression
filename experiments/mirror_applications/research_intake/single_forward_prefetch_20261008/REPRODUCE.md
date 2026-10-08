# Reproducing the current scientific boundary

**Repo preflight:** with Python 3.10+, PyTorch 2.10 CPU and NumPy 2.3.5 installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/mirror_applications/research_intake/single_forward_prefetch_20261008/pilots/sfm001/source/algebra_reference.py
```

The lightweight source reproduces exact shared-forward, output, Attention and gradient identities and the oracle two-buffer *hypothetical* overlap example. It does **not** reproduce the complete wall-clock benchmark, learned readout regression, or hot-fold controls; do not claim that from running it.

**Full original source and all CSV rows:** see the conversation artifact `mirror_single_forward_prefetch_20261008.zip`, generated during this experiment and linked in the corresponding research chat. Within that ZIP:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/test_experiment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/test_folded_control.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/run_experiment.py --phase all --output results
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/run_folded_control.py --phase all --output results
```

- Timing varies with CPU scheduling and clocks; do not demand byte-exact timing CSV replay.
- The main protocol and separately frozen folded amendment are stored beside this runbook. No GPU or data downloads are required for mechanism screens.
- Native GPU expert prefetch (PA434–436), trained useful functions, and TTFT/TPOT/VRAM remain untested.
- Full MA-1175 and MA-1176 scientific statuses remain UNTESTED.

**Proof source:** [full report](REPORT.md). **Recorded summary:** [RESULTS_CORE.csv](RESULTS_CORE.csv). **Original user artifact ZIP:** retrieved from research chat, not from public Git history.
