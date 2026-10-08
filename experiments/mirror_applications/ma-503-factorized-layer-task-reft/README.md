# MA-503 — Factorized layer × task ReFT codes

Status: **FAIL for Mirror-specific attribution and frozen generalization/storage gates**  
Branch: `research/ma-503-factorized-layer-task-reft-20261008`  
Prior art: PA96 (ReFT / LoReFT)

## H — Hypothesis

A rank-two factorization across logical layer and task axes will generalize to eight held-out layer/task combinations in a shared rank-four representation intervention, at relative output RMSE <=.05 and <=.50x a dense pair-code payload when target coefficient interactions follow that factorization.

## Mirror insertion

**Mirror insertion:** this experiment adds layer and task factors whose product forms a hidden-representation code `m_(layer,task)` so that crossed logical interventions can be represented without a separately stored code for every layer/task pair.

The native control uses the identical bilinear layer/task coefficient parameterization. MA-501/502 pause only unchanged static coefficient banks; this experiment tested held-out cross-axis composition. Amendment 1 preserves both initial failed launches and fixes only method-object caching, with protocol, data, optimizer, seeds and metrics unchanged.

## T — Executed protocol

Two frozen worlds (50301/50302); four logical layers × eight task IDs; hidden dimension 24; shared intervention rank four; layer/task factor rank two. Eight pairs where `(layer+task) mod 4 == 0` were held out from basis and factor fitting; 24 pairs remained visible. The full grid was evaluated at private interaction ratios rho=0 and .25. Each visible pair had 64 calibration examples and each pair 128 heldout inputs. The native bilinear coefficient model, dense code table, additive main effects, independent factors on visible pairs and full-matrix oracle were charged by actual uncompressed NPZ bytes.

## D — FAIL

At rho=0, factorized heldout relative RMSE was 0.00215 in seed 50301 but 6.75411 in seed 50302; visible-pair RMSE was 0.00230 and 0.04371 respectively. The factorized payload was 2,568 B; the dense pair table was 2,685 B (only 4.4% larger, missing the <=0.50x gate). The additive main-effects control had heldout error 1.18–1.60; a code table with unseen pairs zero-filled had error 1.0; the full-matrix oracle was exact. The factor model used 200 operation-proxy units/example, compared with 196 for the dense lookup and 576 for a full matrix. Fitting used 3,000 Adam updates and 1.45–2.44 seconds in these small CPU screens.

At rho=.25, factorized heldout error was 0.563 in seed 50301 and 22.515 in seed 50302, despite visible-pair errors of 0.074 and 0.148. The pair-private interaction therefore sharply degrades heldout composition. For both seeds, native bilinear control and Mirror had byte-identical payloads and exactly identical outputs and metrics. Fresh seeds 50311–50313 remain sealed.

## C — Strongest counter-hypothesis

The fixed checkerboard holdout may not identify a rank-two layer/task completion robustly, and one fixed Adam initialization can fit visible pairs while extrapolating poorly. Regardless of that uncertainty, the code is ordinary bilinear factorization and actual NPZ storage barely beats the dense shared-basis table at this small grid.

## U — Still unconfirmed

The source of the seed-50302 generalization failure was not isolated with alternate masks, restarts or optimizers because those were not part of the frozen protocol. No natural ReFT tasks, Transformer behavior, NLL or larger layer/task grid was tested. The result is a negative screen, not a general impossibility proof for factorized ReFT.

## Evidence classification

**Facts:** output metrics, actual payload lengths/hashes and replay are stored in `runs/`, `RESULTS_CORE.csv` and `VERIFICATION.json`; the pre-amendment failed launches are preserved separately.  
**Interpretation:** this candidate misses frozen quality/storage gates and is fully explained by native bilinear coefficients.  
**Hypothesis:** a different holdout design or natural intervention family may alter matrix-completion quality, but would need a newly frozen experiment.
