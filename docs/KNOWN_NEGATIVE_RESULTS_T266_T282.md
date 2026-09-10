# Known negative results — T266-T282

These results are retained because they constrain future FQC claims and search policy.

## 1. Simple whole-model 64x family failed quality

An actual 28M whole-model artifact was serialized at the hard 64x size target and independently decoded. The resulting model quality was severely degraded. Selective correction improved the failed candidate but did not make it a quality-preserving 64x result.

**Consequence:** do not report existence of a 64x-sized file as 64x model compression success. The scientific gate includes task quality.

## 2. More private correction can be worse

For the tested 64x candidate, a selected 960-neuron correction improved quality relative to no correction, while extending support to 2,363 neurons degraded the measured result.

**Consequence:** support size is an optimization variable. Remaining byte budget should not automatically be spent on the largest feasible correction set.

## 3. Lower parameter-space error can worsen task quality

In the 1M affine-baseline experiment, a refit that improved parameter-space reconstruction did not improve functional quality and instead worsened measured KL/NLL relative to the simpler control.

**Consequence:** parameter-space SSE/MSE is diagnostic only. It is not an admissible final quality objective for FQC.

## 4. KL and NLL can disagree

Some tested candidate reallocations improved KL to the original model distribution while worsening ground-truth token NLL.

**Consequence:** KL-only candidate selection can produce false functional wins. Retain both teacher-relative and target-relative metrics, with NLL treated as an independent quality gate.

## 5. Equal logical width does not imply equal final bytes

Two codecs with the same nominal logical width changed ordering after the same reversible outer compression.

**Consequence:** final serialized file size is authoritative. Logical bits, nominal bit width and pre-envelope payload size are insufficient for hard budget claims.

## 6. Activation-aware improvement is not FQC-specific evidence

The strengthened activation-aware non-sharing affine quantizer improved the internal control on the tested checkpoints.

**Consequence:** this method must be used as a stronger baseline. Its gain cannot be attributed to functional sharing, quotienting or Vector Mirror structure.

## 7. Public-prompt pilot is not official validation

The later frozen-candidate pilot used 44 public-prompt-derived inputs. The candidate was not tuned on these inputs, but the input set is small, correlated, and not byte-verified against the upstream YAML or equivalent to the official TinyStories validation corpus.

**Consequence:** pilot improvements are preliminary. They cannot be promoted to official-dataset generalization, SOTA or publication-quality external validation.

## 8. Reproduction is not independent replication

T278-T282 strengthened same-environment deterministic reproduction, separate-process replay, checkpoint-free decode witnesses and integrity tests.

**Consequence:** these results establish engineering reproducibility within the recorded environment. They do not establish independent-machine, independent-implementation, independent-seed, MPS or CUDA replication.
