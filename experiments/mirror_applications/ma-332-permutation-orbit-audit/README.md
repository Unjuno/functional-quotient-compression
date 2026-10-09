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


## A1 results

**H:** hidden permutation and positive scale with inverse compensation preserve a ReLU MLP function and add no distinct logical function; uncompensated Givens changes it.

**T:** A1 used development worlds 33221–33222 and fresh worlds 33231–33233, 4,096 Gaussian inputs/world, a fixed 16-24-8 ReLU MLP, seven methods, no optimizer updates. A0 fresh was quarantined after the serializer dtype defect; see `A0_IMPLEMENTATION_BUG.md`.

**D — FAIL for symmetry-as-functional-multiplicity:** fresh means: baseline 2,528 B; permutation View 2,618 B / output NRMSE 9.03e-8; positive scale plus compensation 2,682 B / 7.18e-8; compensated Givens 2,570 B / 2.70e-8. These are one function within 1e-6. Uncompensated Givens is 2,572 B / NRMSE 0.183; negative scale plus inverse compensation is 2,680 B / 0.899. The extra address bytes buy no new function for exact gauge transforms.

**C:** only one small random MLP and Gaussian input distribution; numerical equality on these probes does not prove equivalence for all inputs, though the permutation and positive-homogeneity cases follow exact ReLU algebra.

**U:** trained networks, other activations/norm layers, task quality and whether useful task-specific functions can be learned from functional Givens-type Views remain untested.

### Fact / interpretation / hypothesis

**Fact:** 21 A1 fresh rows across three worlds; all exact-gauge NRMSE values were below 1e-6. Payload hashes and byte lengths decoded/replayed exactly; metric replay max difference was zero. Three tests pass.

**Interpretation:** permutation and compensated positive scaling are parameter-coordinate changes, not logical functional multiplicity. A functional view must change outputs, and the extra coordinate is paid.

**Hypothesis:** this audit rule should be applied before counting hidden-unit symmetry views as experts in larger architectures.
