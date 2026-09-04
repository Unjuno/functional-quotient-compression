# Mirror scaling replication protocol (T185–T190)

## Scope

These results are methodology evidence for testing the Functional Mirror / FQC scaling hypothesis. They do not add a new real-model compression result.

## Core boundary: conditional family trend vs population scaling law

With one checkpoint at each TinyStories scale, block/fold resampling can quantify measurement uncertainty **conditional on those exact checkpoints**. It cannot turn those four checkpoints into independent model-scale replicas.

Synthetic calibration in T188 shows the distinction clearly: increasing block seeds keeps the fixed-family null false-positive rate near the nominal 2.5% and raises power for a fixed family slope, but under a population null with checkpoint-to-checkpoint variation the false-positive rate rises as more block repeats lock onto a random realized family slope.

Therefore the current 1M/3M/8M/28M checkpoints are a **discovery / conditional-family set**, not a population scaling-law sample.

## Pseudoreplication boundary

T187 shows that layers are not independent model replicas. When checkpoint-level variance is present, treating eight layers from each of four checkpoints as 32 independent scale observations can strongly inflate false positives. The same restriction applies to blocks, fold rotations, pair prices, and coalition candidates.

The inference unit for a population scaling law is an independently trained model replica.

## Discovery vs confirmation

T189 tests selection bias explicitly. After a negative trend is discovered in the original four checkpoints, confirm population scaling using **new independently trained replicas only**. Reusing the selected discovery checkpoints inside the confirmatory regression carries selection bias into stage 2 and materially inflates false positives in the synthetic design.

Recommended evidence lanes:

1. **Discovery / conditional lane** — existing four checkpoints; matched-support fold-rotated geometry diagnostic, cross-fit private geometry metrics, and block/fold repeats for conditional measurement uncertainty.
2. **Confirmation / population lane** — entirely new independently trained replicas; model replica is the inference unit.
3. **Codec lane** — full-model emitted-byte and functional replay evidence, analyzed separately from the scaling-law inference.

## Replica allocation

T185–T186 stress replica allocation under a fixed training-cost proxy. For detecting a linear log-width slope, endpoint replicas have high information value, so equal replica counts are not generally cost-optimal. This is a slope-power result only; intermediate scales remain necessary for curvature, crossover, and plateau diagnostics.

A practical confirmatory campaign should retain multiple largest-scale replicas even when they are expensive. Designs with only one or two largest-scale models weakly identify largest-scale variance even if the linear-slope power calculation can be compensated by many cheaper replicas elsewhere.

## Sequential confirmation

T190 tests a compute-saving confirmation design using only new independent replicas. In the synthetic variance/effect setting, a useful calibrated example is:

- start with 4 new model replicas per scale;
- early efficacy if one-sided `p < 0.005`;
- futility if `p > 0.5`;
- otherwise expand to 8 replicas per scale;
- final threshold `p < 0.0225`.

This preserved nearly the power of fixed 8-replica confirmation while reducing expected replica count. These numerical thresholds are **not universal** and must be recalibrated using real observed between-model and within-model variance before a real confirmatory campaign.

## Scientific interpretation

The current evidence can support a statement of the form:

> Within the observed TinyStories checkpoint family, shared functional representation shows a scale-associated trend under the tested protocol.

It cannot yet support the stronger statement:

> The model population obeys a general mirror-compressibility scaling law.

That stronger claim requires independent training replicas across scale and a pre-specified confirmatory analysis.