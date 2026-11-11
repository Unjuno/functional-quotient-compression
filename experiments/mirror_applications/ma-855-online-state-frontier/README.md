# MA-855 — common-task online state frontier

Amendment 01 replaces the initial orthogonal-support prototype before any experiment run. Astra review found that identity support would make Hebbian and Delta updates exact and that a supplied teacher basis would leak task structure.

H: on a common linear task family with nonorthogonal Gaussian support, compact support-adapted state can retain query quality with fewer bytes than full matrix updates. Mirror-specificity must beat CAVIA-style context and native low-rank Delta controls.

A separate train-task set supplies noisy-free support observations from which the shared mean model and low-rank decoder basis are fitted. Evaluation teachers are independent hidden coefficient and residual draws; the learner receives only shared checkpoint plus support pairs. Compare shared mean, full Hebb, full normalized Delta, support-gradient context, projected Delta state, native low-rank Delta, and support-only ridge.

D=8 input, O=4 output, rank-4 decoder fitted from 128 train tasks. Each development/fresh task has 32 random Gaussian support and 128 independent query examples. Development seed 29 tunes method rates; fresh seeds 73/89 are locked. Report pretraining examples separately from per-episode support, updates, actual serialized model plus state bank bytes, online write compute, query compute and wall time. No meta-training is claimed.
