# MA-260 — BatchEnsemble rank-one vs Mirror task ensemble views

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE  
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`

## Hypothesis

H: On a task family whose member classifiers differ by a low-dimensional orthogonal input rotation, storing one shared classifier plus one scalar rotation coordinate per member preserves member accuracy and ensemble diversity at fewer bytes than BatchEnsemble's member rank-one factors.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-member angle `m_t` to a shared binary linear classifier so that rotated task decision boundaries can be expressed without storing a full member classifier vector.

- Native method: BatchEnsemble shares a weight and stores member-specific rank-one fast-weight factors.
- Exact insertion: input-side 2D rotation of one shared classifier vector.
- `m_t`: persistent task address, one angle per member.
- Cheapest ordinary comparison: BatchEnsemble rank-one member input/output factors.
- Physical object: one binary linear classifier vector.
- Logical objects: six task classifiers.

## Prior-art delta

PA17 establishes shared weights with member rank-one fast weights. This screen tests a lower-dimensional orthogonal address against those member factors. It does not claim ensemble learning itself as a Mirror contribution.

## Protocol

Synthetic binary classification, six task members, input dimension 8. In the aligned condition, each task's teacher separator is a different rotation of a common vector in a fixed 2D plane. In the unaligned stress condition, each task separator is independently sampled. For each task, fit an ordinary logistic separator on 512 examples, then evaluate its encoded representation on 1024 held-out examples. This is a post-fit representation screen, not ensemble training.

Controls: one shared classifier, independent fitted classifiers, BatchEnsemble rank-one factors, and Mirror rotation coordinates. Seeds 13 and 29 are development; fresh seeds 103, 223, 317, 409 are frozen in the protocol. No hyperparameter tuning from fresh results.

## Gates

PASS: aligned condition preserves independent accuracy within 1 percentage point and pairwise disagreement within 2 points while using <=25% of BatchEnsemble bytes on at least 3/4 fresh seeds; unaligned condition is reported as a scope test.

FAIL: aligned Mirror loses >5 accuracy points on at least 3/4 seeds, or its actual bytes exceed BatchEnsemble while quality is within 1 point.

NOT ESTABLISHED: serialization or task split cannot be reproduced.

## Storage / compute

Actual uncompressed NumPy NPZ payload bytes include all learned classifiers/factors/codes and biases. Fit examples, wall time, inference MAC proxy, accuracy, ECE and member disagreement are recorded. Runtime comparisons are excluded because decoding/rotation kernels are not optimized.

## Results

See `RESULTS_CORE.csv`, `artifacts/`, and `VERIFICATION.json`.

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic linear classification only; no neural ensemble, calibration deployment, language, or capacity claim.

## Results

FACT: On four fresh seeds in the rotation-aligned condition, independent and BatchEnsemble rank-one achieved mean accuracy 0.8265 and ECE 0.0254. Mirror achieved accuracy 0.8302 and ECE 0.0226. Pairwise member disagreement was 0.502 for independent/BatchEnsemble and 0.501 for Mirror. Actual payloads were 926 bytes independent, 1,226 bytes BatchEnsemble, and 890 bytes Mirror. Mirror saved 27.4% against the implemented BatchEnsemble state and 3.9% against independent members, well short of the predeclared <=25% byte ratio to BatchEnsemble. In the independent-random stress condition, Mirror accuracy was 0.6018 versus 0.8332 for independent/BatchEnsemble. Results were consistent with two development seeds.

INTERPRETATION: The aligned case indicates the rotation coordinate can compact a known orbit, but the registered storage gate failed. The unaligned task family collapses badly. In this single-output linear toy, the BatchEnsemble factors do not provide a favorable storage baseline; ordinary independent weights are already smaller than the implemented BatchEnsemble state.

HYPOTHESIS: Low-dimensional orthogonal task families can use angle-coded views, but the benefit depends on the task family and on the size of the shared network relative to member codes.

COUNTER-HYPOTHESIS: The observed advantage is a consequence of a teacher designed as a rotation orbit and a weakly representative, degenerate BatchEnsemble linear control. A deep network with standard vectorized rank-one factors may dominate.

UNCONFIRMED: standard deep-network BatchEnsemble, trained neural ensembles, calibration under distribution shift, throughput, and fixed-byte convergence. This is a narrow mechanism FAIL, not a general verdict on Mirror ensembles.

Decision: FAIL the predeclared useful storage gate; preserve the aligned existence case and unaligned collapse as scoped observations.
