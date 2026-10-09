# MA-332 — Mirror permutation-orbit audit

Status: SCREENING. Prior art: PA37 Git Re-Basin; PA47 monomial weight-space symmetries.

## H — Hypothesis

For a two-layer ReLU network, hidden-unit permutations and positive hidden-unit rescalings with inverse outgoing compensation preserve the function to numerical precision; these Views therefore add zero functional multiplicity. An uncompensated hidden Givens rotation changes the function. The audit falsifies any capacity claim that counts exact gauge Views as new functions.

## Mirror insertion

> **Mirror insertion:** this audit adds a discrete hidden-unit address `m` to the hidden activation/parameter coordinates, distinguishing function-preserving gauge relabelings from functional changes before counting logical models.

The shared physical object is a fixed two-layer ReLU MLP. `m` selects a permutation, positive diagonal rescaling, or hidden Givens rotation. The cheapest control is direct reindexing/compensated parameter transformation; independent weights are the upper reference. No learned task specialization is claimed.

## T — Protocol

A 16-input, 24-hidden, 8-output ReLU MLP with fixed random weights is evaluated on 4,096 deterministic Gaussian inputs for three fresh worlds (33211–33213), plus development worlds (33201–33202). Compare baseline, exact permutation, positive scaling with inverse compensation, negative scaling with inverse compensation, uncompensated Givens activation view, and compensated Givens parameter transform. Serialize the shared weights and each paid view code separately; report exact actual bytes and output NRMSE. Zero optimizer updates; report operations and isolated wall time.

## Gates

**PASS:** exact symmetry classes show output NRMSE <=1e-6 and functional distinct count per symmetry class remains 1; uncompensated Givens has NRMSE >1e-3. **FAIL:** an asserted exact symmetry changes outputs beyond tolerance or an asserted functional view is numerically identical. The result is an audit/control, not a compression win.

## C / U

ReLU admits hidden permutations and positive diagonal rescalings with inverse outgoing compensation. Negative scaling is not generally a symmetry. The result covers this MLP and transformations only; no transformer, trained network, data task, or deployment inference is tested.
