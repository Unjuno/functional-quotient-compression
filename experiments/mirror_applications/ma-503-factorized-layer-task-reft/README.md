# MA-503 — Factorized layer × task ReFT codes

Status: SCREENING; protocol frozen before development; amendment 1 is a cache-only source correction before canonical reruns  
Branch: `research/ma-503-factorized-layer-task-reft-20261008`  
Prior art: PA96 (ReFT / LoReFT)

## H — Hypothesis

A rank-two factorization across logical layer and task axes will generalize to eight held-out layer/task combinations in a shared rank-four representation intervention, at relative output RMSE <=.05 and <=.50x a dense pair-code payload when target coefficient interactions follow that factorization.

**Mirror insertion:** this experiment adds layer and task factors whose product forms a hidden-representation code `m_(layer,task)` so that crossed logical interventions can be represented without a separately stored code for every layer/task pair.

The principal native control is the identical bilinear layer/task coefficient parameterization. MA-501/502 pause only unchanged static coefficient banks; this experiment tests held-out cross-axis composition. Amendment 1 preserves the two initial failed launches and corrects only method-object caching; it leaves protocol, seeds, fit and metrics unchanged.
