# Federated Mirror code family diagnostic after MA-341 and MA-342

Date: 2026-10-09 UTC

## Trigger

The worker stop rule pauses a family after two consecutive candidates fail for the same demonstrated structural cause. MA-341 and MA-342 are consecutive P0 federated personalization screens and both miss the Mirror-specific byte margin for the same reason.

## Fact

- MA-341: held-out synthetic client MSE was 1.016e-4 for phase Mirror at 516 B and direct two-coefficient basis at 556 B. pFedHN-style generated full predictors used 4,783 B at MSE 1.184e-4. Mirror's 7.2% saving over simple coefficients missed the 10% gate.
- MA-342: held-out synthetic client MSE was 1.0014e-4 for phase Mirror at 516 B and direct two-coefficient basis at 556 B. HyperLoRA-style factor generator used 5,048 B at MSE 1.0298e-4. Again the code saved 7.2% over the simple control, below the 10% gate.
- Both teachers are planted on shared circular low-dimensional orbits. Neither run used real federated optimization, natural client adapters or privacy/communication rounds.

## Interpretation

The large savings relative to generated full predictors/adapters come from a shared low-dimensional basis. Encoding two coefficients as one phase reduces per-client address bytes by only 4 B and contributes under 10% to total payload. The direct coefficient control realizes the same function family and nearly the same total storage. The comparison does not justify a Mirror-specific federated claim.

## Redesign condition

Pause MA-343..348 until a candidate names a natural task bank and a plausible source of materially smaller client state than ordinary coefficients/embeddings, and includes a meaningful private-residual boundary. Any resumed work must compare against the native HyperLoRA product-space generator, count communication rounds and full server/client state, and evaluate real held-out clients.

## Queue decision

Pause `Federated personalization` under the family stop rule. Continue with MA-349, the independent distillation/ensemble/uncertainty family.
