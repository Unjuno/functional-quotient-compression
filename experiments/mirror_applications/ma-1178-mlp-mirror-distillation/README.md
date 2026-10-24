# MA-1178 — Pretrained MLP function -> shared Mirror distillation

Date 2026-10-24 JST. **Pre-registered prior to training.** This study answers a functional question: when ordinary nonlinear MLPs already implement 3 distinct teacher functions, how well can a much smaller shared-core Mirror MLP distill those functions? This is not a language model, a causal attention benchmark, or evidence that a Mirror generates independent degrees of freedom.

## Design

- First train 4→16→4 ordinary tanh MLP teachers on nonlinear analytic functions and freeze them.
- **Mirror-related teachers:** starting from one pretrained base teacher, construct three *ordinary dense MLP* networks whose first/second-layer matrices are conjugated by known Householder transforms. This deliberately creates a representable favorable teacher family; student does NOT get the teacher core or true Mirror directions.
- **FiLM-related teachers:** similarly construct 3 ordinary dense teachers by learned-elementwise-style hidden gain transforms of the pretrained base teacher. This is a counterexample world favorable to a non-Mirror formulation.
- **Independent teachers:** train three separate ordinary tanh MLPs on different nonlinear analytic functions. No assumed shared function geometry.
- Student has common up/down matrices (physical shared coefficients), with either 3 learnable Householder directions, 3 equally sized FiLM vectors, or no codes. A matched-physical-size independent-MLP student is the strong native comparator.
- Student widths H=8/16/24. Mirror/FiLM stored floats = `12H+4`, including all hidden biases, output biases, and three H-dimensional codes. Native independent students have H'=3/7/10 and 3*(9H'+4) floats. Teacher 3×(9*16+4)=444 floats.
- Minimize MSE in the teacher's **function outputs**, using train-only squared teacher norm to normalize. This avoids Cross-Entropy confounds. Objective is functional imitation, not supervised accuracy of the external analytic targets.
- First prove that an oracle shared Mirror student H=16 with true pretrained weights/directions reproduces the three aligned teachers exactly in floating point. This is only a representability witness; separately train all students from random initial weights and random codes.

## Frozen evaluation

Five independent teacher/task seeds; student train/dev/fresh/OOD sets disjoint by seed; all hyperparameters fixed by `PROTOCOL.json`. Fresh and OOD inputs are generated **after** all 180 final student checkpoints are hashed. Select best student checkpoint using development set alone. Report Fresh/OOD normalized MSE, worst-rule NMSE, actual complete inference torch payload bytes, trained scalar count, training wall clock, forward CPU microbenchmark. Preserve all failures and all 180 checkpoints.

## Claims boundary

The aligned teachers are **constructed to be representable by this Mirror family**; success there is not a generic Mirror advantage. A true Mirror-specific claim requires beating exact-parameter FiLM and comparable-parameter independent MLPs at same width/budget, preferably on independently constructed teachers. The Householder map itself has O(H) fused application; generic learned dense transform is not free.

## Run

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v`

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run.py --out .`

`python source/analyze.py`
