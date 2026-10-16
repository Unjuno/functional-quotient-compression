# GIC-001 Implementation Plan

> Execution: inline, in this container; approved continuation of the gradient-interference audit.

**Goal:** Separate shared-function representation limits, finite optimization gaps and the value of conflict-guided private allocation.
**Architecture:** An exact quadratic sanity check plus trainable cubic spline banks, rank-SVD optima, fixed-budget interventions and actual inference codecs.
**Tech Stack:** Python 3.13, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10 CPU for independent gradient checks, pytest.
**Spec:** PROTOCOL.json. Existing scientific records and sealed AMA seeds remain untouched.

## Global constraints
All computation in /mnt/data/gic001. Container cannot resolve github.com; read/write repository through its connected API. No remote compute, no main merge, no invented MA IDs. New study, not reproduction of inaccessible historical checkpoints.

## Review focus
Zero gradients; nonzero gradient conflicts at compatible optima; private allocation must preserve the current function; exact byte accounting; no data or RNG sharing between training and audit.

- [ ] Write and observe failing unit tests for the spline basis, PCGrad, gradients, rank oracles and codec.
- [ ] Implement gic.py; verify against independent autograd and least squares.
- [ ] Lock source and protocol hashes BEFORE any development training.
- [ ] Run one development world in all four regimes; no efficacy-based tuning.
- [ ] Run 5 fresh worlds x 3 initializations x 4 regimes x 11 methods (660 fits).
- [ ] Reload every payload, regenerate data and independently recompute metrics; exact replay one complete training case.
- [ ] Benchmark separately; analyze at world rather than initialization level.
- [ ] Publish code, protocol, summary and verification to a NEW research branch after local verification; retain full evidence ZIP here.
