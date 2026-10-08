# MA-346 — Federated shared/private residual frontier

Status: **PROMISING for a narrow synthetic shared/private storage frontier; no Mirror-specific advantage**
Dedicated branch: `research/ma-346-federated-shared-private-mirror-20261008`
Prior art: PA38 HyperLoRA; PA40 pFedHN; FedRep-style local-private split as a direct baseline.

## H — Hypothesis

As clients move away from a shared two-direction task family, a phase-addressed shared model plus rank-1 private residuals should preserve quality at lower total bytes and communication than dense private heads. The heterogeneity sweep measures where private state starts to dominate. Direct coefficients and ordinary scalar phase determine whether the benefit is Mirror-specific.

## T — Test

Sixty-four synthetic 16×16 client functions have a shared two-direction phase family. Outlier fractions were swept at 0%, 25%, 50%, 75%, and 100%; each outlier adds an independent rank-1 residual. Controls: independent full matrices, one global shared model, phase with/without private factors, direct coefficient plus rank-1 residual, native scalar phase, and a FedRep-style dense private residual. Each client was evaluated on 512 held-out inputs. Dev seeds 34601/34602; fresh seeds 34611/34612/34613.

All ZIP/NPY FP32 inference payloads were reloaded before evaluation. Client phases and outlier assignments were planted; no optimizer updates or local adaptation were performed. The 25% and 50% rates were preregistered as the primary region after the development sweep; all rates remain reported.

## D — Decision

**PROMISING for the shared/private frontier at 25–50% synthetic heterogeneity.** On all three fresh worlds, private-enabled phase had nMSE 0 for both aligned and outlier clients. At 25% heterogeneity, Mirror used 6,739B vs 20,243B FedRep dense-private (66.7% fewer bytes); at 50%, 8,639B vs 35,696B (75.8% fewer). Client communication proxies were 2,336B and 4,416B versus 16,928B and 33,344B. The private rank-1 payload increases with outlier count; no-private outlier nMSE was approximately 0.0035–0.0037, above the 1e-5 quality gate.

At 0% heterogeneity, Mirror was only 2.1% smaller than the dense-private control; this endpoint missed the 20% gate. At 100%, Mirror still used 12,451B versus 66,520B dense-private and 61,228B independent full matrices, but the shared basis had little remaining work to do. The direct coefficient control averaged 269B more than phase, below the 10% Mirror-specific margin. More decisively, native scalar phase matched Mirror byte-for-byte across every seed and rate. No Mirror-specific advantage is claimed.

## Fact / Interpretation / Hypothesis

**Fact:** In the primary 25–50% fresh region, private-enabled quality was exact and payload was 66.7–75.8% below dense private. Private state grows with the outlier fraction. Native scalar phase payload hashes equal Mirror hashes.

**Interpretation:** A shared/private factorization can substantially beat dense local private matrices when only some clients need rank-1 residuals. The gain comes from low-rank residual storage and the known common phase family, not a uniquely Mirror implementation.

**Hypothesis:** Real client banks may show a similar private-state threshold, but trained adaptation, client drift and communication rounds could change the frontier.

## C — Strongest counter-hypothesis

The benchmark plants both the common two-direction basis and rank-1 outlier residuals. A direct shared basis plus scalar phases reproduces Mirror exactly, and FedRep is represented by a dense rather than learned low-rank private-head control.

## U — Unconfirmed

No trained federated models, real client/task heterogeneity, optimizer adaptation, unseen client phases, privacy, bandwidth latency, or language-model evidence. Private residuals are oracle-known after fitting.

## Reproduction

```bash
python experiments/mirror_applications/ma-346-federated-shared-private-mirror/source/run.py
python -m pytest -q experiments/mirror_applications/ma-346-federated-shared-private-mirror/tests
python experiments/mirror_applications/ma-346-federated-shared-private-mirror/source/verify.py
```
