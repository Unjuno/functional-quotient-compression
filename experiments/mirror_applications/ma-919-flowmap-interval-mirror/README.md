# MA-919 — two-time flow-map interval Mirror code

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE

## Hypothesis

A shared map addressed by start time and duration can improve held-out endpoint map or semigroup composition quality per byte/compute over native start/end conditioning.

## Frozen amendment 01

The official PA260 paper has no author-released FMM code found in bounded searches; its paper links an unrelated baseline architecture repository and reports image experiments. To make bounded progress without pretending to reproduce PA260, this package preregisters an analytic 2D linear-flow mechanism screen. Mirror coordinate `m=(s, t-s)` is compared with native `(s,t)` through identical shared MLPs on three fresh linear systems. This coordinate is an invertible reparameterization and thus has a strong native alias control.

Metrics: held-out exact-map MSE, semigroup composition MSE, actual serialized state-dict bytes, MAC proxy, and wall time. No FID or image generation claim.

## Result interpretation

Only a mechanism screen. A win cannot establish PA260 image quality or general generative capacity; a tie/lose falsifies this simple interval coordinate variant.
