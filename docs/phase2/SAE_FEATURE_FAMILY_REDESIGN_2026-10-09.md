# SAE feature-view family redesign trigger — 2026-10-09

## Trigger

MA-526 and MA-527 are consecutive screens in the pretrained SAE feature-view family and both failed their frozen development gates for the same structural reason: the tested SAE feature representation does not preserve the function-vector task effect well enough to justify feature-space compression.

## Evidence

- **MA-526:** shared 64-atom pool loses 1.463/1.567 gold-logprob nats versus explicit FVs; misses seed quality and the global OMP margin; the SAE adds 4,172,903 B to standalone deployment. Its registered fresh seeds 52611–52613 were not run.
- **MA-527:** fixed Givens over a 32-atom SAE bank loses 2.094/1.309 nats versus same-scale explicit FVs; one dev seed also misses accuracy tolerance and the Mirror code is dominated by a simple control. The 3,474 B incremental code is much smaller than explicit FVs, but standalone deployment is 4,173,123 B larger because the SAE is charged. Fresh causal metrics were not computed. Task IDs 14–15 were exposed by dev extraction, and fresh seeds 52711–52713 were not run.

## Ruling

Pause MA-528 and MA-530 pending a family redesign. Their current premises reuse the same SAE feature object and feature/task alignment assumption. Do not open held-out data for these candidates under that unchanged representation assumption.

This pause does not cover the separate transcoder family (MA-533/534): a trained sparse MLP approximation is a different physical object and must still be compared to its own native controls.

## Resume gates

Before resuming MA-528 or MA-530, preregister a concrete change that addresses SAE/task alignment (for example, feature selection from training-only causal gradients, a task-aligned learned dictionary, or a nonlinear feature decoder). The revised protocol must include a development-only alignment audit, an explicit-FV and native sparse-coding control, full dictionary bytes in the standalone frontier, and new sealed held-out task identities. A third unchanged SAE-bank variant is not sufficient.

## Classification

- **Facts:** the two development screens missed their preregistered causal quality gates; no fresh-seed causal audit was run. Task IDs 14–15 were inadvertently extracted during development and cannot be used as sealed audit identities.
- **Interpretation:** feature-bank code bytes are compressible, but the tested pretrained SAE coordinates discard task-relevant function-vector directions, and full-system SAE storage erases the deployment saving.
- **Hypothesis:** task-aligned feature bases or nonlinear decoders may alter this frontier; the current evidence does not test them.
