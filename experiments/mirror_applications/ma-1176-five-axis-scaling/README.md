# MA-1176: Microbenchmark to isolate five scaling axes

Focus: **Loop / Mirror / physical parameter / data information / architecture-search budgets**. Minimal 4-dimensional synthetic regression and three symbolic rules, students 32–288 learned scalars. This is a diagnostic, not a new Transformer or LLM training claim.

The independent variable and claimed unit of each axis differs:

- Loop = extra sequential computation with the *same* learnable coefficients; examine whether expensive training-time refinement can be distilled to one deploy step.
- Mirror = multiple rule-dependent nonlinear functions sharing physical coefficients; compare exactly parameter-matched FiLM/simple gates, and independent per-rule MLPs.
- Parameter = actually trained FP32 scalars and serialized bytes; fixed training data and update count.
- Data = number of unique examples, rule distribution, poisoned labels and oracle-only filtering; fixed number of optimization steps.
- Search = number of alternative trained candidate structures/initializations; include all discarded training compute and dev-only selection.

**Pre-reg**: see `PROTOCOL.json`. Fresh/OOD are sealed until training checkpoints are hashed. Report split-conditional errors, bytes, training time and a forward MAC proxy separately. Synthetic aligned-Mirror teachers are favorable to the Mirror family and never sufficient on their own for a superiority claim. The independent teacher world is mandatory. Do not assume a monotonic scaling relationship or any transfer to pretrained attention.

Operational notes: CPU FP32, 1 numerical thread, PyTorch 2.10. `python source/run.py --phase train` followed by `--phase eval`; `python -m unittest discover -s tests -v`; `python source/analyze.py` once eval is complete. `--phase train` never generates fresh examples. `--phase eval` verifies a sealed manifest first.