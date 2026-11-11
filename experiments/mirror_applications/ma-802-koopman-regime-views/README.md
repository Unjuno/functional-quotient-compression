# MA-802 — Koopman latent regime Views

Status: **FAIL for Mirror-specific attribution** (synthetic storage/quality mechanism passes against independent operators)

## Prior-art boundary

PA214 motivates shared encoded coordinates with regime-conditioned latent dynamics and requires stability audits. This screen uses a known orthogonal nonlinear observation map `x=tanh(Qz)` and an analytically invertible encoder. It does not learn a Deep Koopman embedding from nonlinear data.

## Fact

Seed 80201; 8D latent state; nine regimes on a 3×3 factor grid; two diagonal regimes held out from transition and basis fitting; horizon 40; 128 train sequences per observed regime and 128 test sequences/regime; zero optimizer updates. Independent regime matrices serialize to 3,565 B with heldout rollout nRMSE 1.01e-7. Shared operator without a regime code uses 1,636 B but has heldout rollout nRMSE .05152. Shared core plus rank-2 Views uses 2,607 B, heldout rollout nRMSE 1.06e-7, and max spectral radius .536, reducing bytes 26.9% versus independent. The native conditional low-rank control has identical payload hash `490e19f2461e14f8cd7b655b393a0c5b76ff93740da5e7541daedb272e8b227e` and matching metrics. Active MAC proxy is 320/step versus 192 for independent; all payloads roundtrip exactly.

## Interpretation

Compact latent regime coordinates recover the two heldout compositions while maintaining stable long-horizon forecasts in this designed system. They improve storage and quality relative to independent full operators, but require 1.67× active MACs versus the independent control. The exact native low-rank conditional parameterization reproduces every byte and output, so the result is not Mirror-specific.

## H / T / D / C / U

- **H:** Shared invertible nonlinear coordinates plus regime Views recover heldout Koopman compositions with stable horizon-40 forecasts and lower complete storage than independent regime operators.
- **T:** Seed 80201; known orthogonal tanh conjugacy; 8D stable latent systems; 9 regimes, 2 held out; rank-2 additive latent operators; independent, shared-only, View and native low-rank controls. Measured full NPZ bytes/hash, one-step and rollout error, spectral radius, compute proxies, sequence counts, CPU rollout time and payload replay.
- **D:** **FAIL** Mirror-specific attribution because the native low-rank control exactly aliases the View. The synthetic storage/quality/stability gates versus independent operators pass; compute increases 1.67×.
- **C:** The teacher is exactly a low-rank additive family of latent transition matrices; ordinary conditional low-rank dynamics provides the same basis and regime coefficients.
- **U:** Learned encoders, data-driven nonlinear Koopman discovery, nonlinear latent dynamics, fresh regimes, control tasks, training effects, GPU/fused runtime and private residual frontier.

Fresh remained sealed after exact native alias. No trained Deep Koopman or general nonlinear forecasting claim is made.
