# MA-849 deferred screening note

MA-849 is P1; it was started before confirming the priority boundary and is deferred in favor of eligible P0 MA-851. The development screen selected rate 0.0612766 and multiplier 0.975 on seed 17.

A fresh invocation for seeds 37/59 was accidentally made before deferral. Its console output was truncated and no metrics were retained in RESULTS_CORE.csv. Those seeds are considered exposed and must not be reused. No tuning followed. Do not infer any fresh outcome from the partial console output.

Astra review recommends replacing this incomplete prototype with a support-only normalized delta learner: teacher y=A x, but learner gets only support x,y and query x; W starts at zero; updates use residuals and support-only covariance-derived SPD preconditioning. Compare fixed NLMS, analytic whitening NLMS, native diagonal/Cholesky context generators, Mirror coordinates at matched bytes, and ridge support-fit upper control. Never pass teacher A or task covariance to the learner. A future MA-849 run needs an amended protocol and new fresh seeds.

This note is an execution/provenance disclosure, not an MA-849 result. Registry status remains UNTESTED until a valid protocol is completed.
