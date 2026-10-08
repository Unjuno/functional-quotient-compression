# MA-411 status: PROMISING (aligned feasibility only)

## H — falsifiable hypothesis

Sparse Mirror coordinates on a fixed Hadamard basis recover at least 90% of dense residual held-out operator fit quality with under 40% of its payload bytes and condition number at most 2.

## T — executed

Four-context, dimension-32 synthetic operator regression. The teacher is a normalized Hadamard plus eight small residual values on one shared, preregistered support. Compared canonical-only, sparse Mirror raw, sparse Mirror with a proven Frobenius bound guaranteeing condition number <=2, context rank-2 residual, dense residual, and independent full matrices. 200 AdamW updates × batch 128; 2 development worlds × 3 seeds selected LR; 3 fresh worlds × 3 seeds. Selected LR was 0.003 for canonical/rank2/dense/independent and 0.01 for both sparse modes. Actual serialized state includes support indices and all learned values/factors.

## D — PROMISING, narrowly

Fresh mean normalized RMSE: canonical 0.016454; sparse raw 0.000005; sparse conditioned 0.000005; rank-2 0.009389; dense residual 0.004641; independent matrices 0.000297. Sparse payload 6,369B versus 22,625B dense residual (28.2%) and independent (28.2%). Max fresh condition number was 1.08961 for raw Mirror and 1.08961 for conditioned Mirror, both below 2. Sparse Mirror MAC proxy was 1,032/example vs 2,048 dense residual. The explicit condition bound did not activate because the fitted sparse code stayed below its radius; therefore conditioning showed safety compatibility, not a quality gain.

## C — strongest counter-hypothesis

This task gives every context the same known sparse support and generates targets from exactly that parameterization. It is an aligned feasibility test; the result may disappear when supports differ or must be discovered from natural operators. The independent upper control also fit the teacher well at higher bytes, so this is a structured compression result, not broader function-capacity evidence.

## U — unresolved

Unknown support recovery, unrelated sparse supports across contexts, learned/nonlinear transforms, natural neural-network operators, runtime in a fused sparse kernel, and language-model quality remain untested. Sparse refinement training was slower than canonical-only due reconstruction and sparse scatter overhead.
