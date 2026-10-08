# MA-643 — GaLore gradient subspace with Mirror coordinates

Status: NOT ESTABLISHED — blocked before a valid experiment
Evidence lane: MECHANISM / OPTIMIZER MEMORY / COMPUTE
Base commit: `c97bf7d` (`research/mirror-application-worker-ready-20261007`)

## H — hypothesis

On a synthetic task whose useful gradients occupy a known low-dimensional subspace, a small Mirror code that selects among shared gradient atoms may reduce optimizer-state bytes relative to ordinary GaLore and preserve update quality; a byte-matched ordinary coefficient vector may erase any Mirror-specific advantage. The CPU screen tests only this mechanism and does not establish a GaLore training result.

## Mirror insertion

> **Mirror insertion:** replace the full per-parameter Adam moments of GaLore's projected gradient coordinates with a shared gradient-atom basis plus a small per-run Mirror coordinate `m` that addresses active atoms.

GaLore's periodic SVD projector remains the native baseline. For a small controlled quadratic, define aligned gradient atoms from development tasks, then compare (a) full Adam moments, (b) GaLore projected Adam moments, (c) hard-tied shared update atoms, (d) Mirror-addressed atoms, (e) ordinary dense coefficient codes, and (f) unrestricted full-gradient upper control. The Mirror mechanism is kept separate from GaLore's SVD refresh so the address contribution can be isolated.

## Environment limitation

The provided container has NumPy/SciPy but no PyTorch, JAX, TensorFlow, CUDA runtime, or visible GPU. Therefore the intended GaLore implementation/training comparison cannot run here. Any NumPy result is labeled CPU algebraic mechanism evidence only; no optimizer memory, training loss, GPU compute, or GaLore reproduction claim will be made from it.

## Prior-art delta

PA142 establishes periodic low-rank projection of full gradients and projected optimizer moments. MA-643 asks whether a further compact code can address reusable update directions beyond that native projection. The mandatory falsifier is an ordinary coefficient/low-rank update control at matched serialized state and update FLOPs.

## Execution ruling

The registered hypothesis requires a meaningful GaLore projection/refresh and optimizer-state comparison. This container lacks PyTorch, JAX, TensorFlow, CUDA, and visible GPU hardware. A tiny NumPy quadratic prototype was explored during development, but its matrix shape and shared-basis byte amortization did not model GaLore's projection economics and could not answer the registered question. It was rejected before any fresh run; no experimental conclusion is drawn from it.

## H / T / D / C / U

H: compact code-indexed shared gradient atoms can retain updates on aligned quadratic tasks while shrinking optimizer state; the same information may be represented by ordinary coefficients.
T: environment capability check only. PyTorch/JAX/TensorFlow/CUDA/GPU absent. A development-only NumPy prototype was rejected as an invalid proxy for GaLore; fresh seeds were not opened.
D: NOT ESTABLISHED. The registered learning/optimizer-memory question was not executed.
C: ordinary GaLore's own projected Adam state or a standard low-rank coefficient vector can capture all benefit without Mirror-specific geometry.
U: actual GaLore implementation, periodic projector refresh, end-to-end training, GPU memory/throughput, natural neural gradients, and optimizer-state checkpoint serialization.

## Fact / Interpretation / Hypothesis

FACT: the current runtime has no supported autodiff framework or GPU; see VERIFICATION.json. No fresh experiment was run.
INTERPRETATION: this candidate is blocked in the current environment; the attempted tiny CPU prototype did not match the registered method and is not evidence.
HYPOTHESIS: a reusable address over update atoms may help only when gradient subspaces recur across tasks or refresh windows.
BOUNDARY: synthetic quadratics, fixed shared atoms, and CPU NumPy operations only.
