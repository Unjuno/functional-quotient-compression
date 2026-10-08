# MA-341 — Client Mirror phase vs pFedHN-style full-model generation

Status: **FAIL for the frozen Mirror-specific margin at the development gate; fresh seeds sealed**  
Dedicated branch: `research/ma-341-federated-mirror-vs-pfedhn-20261008`  
Prior art: PA40, Personalized Federated Learning using Hypernetworks (pFedHN).

## H — Hypothesis

A one-scalar client phase over a shared basis can represent personalized client models on a known rank-2 orbit, generalize to clients without local samples, and reduce total server-plus-client payload versus a full-model hypernetwork. A native scalar-latent control tests whether any gain is Mirror-specific. An off-orbit client should need private state.

## T — Test

Synthetic federated linear regression: 8 input and 4 output dimensions, 64 clients on a sinusoidal rank-2 model family, 48 clients with 32 support examples each, and 16 clients held out from local fitting. One additional off-orbit client has 32 local support examples. Each method was evaluated on 512 held-out inputs per client. The server fitted a linear hypernetwork from the observed client models and two-dimensional client descriptors. Mirror stores the same generator with one phase scalar per client. Controls include independent full models, a shared FedAvg matrix, pFedHN-style generation with/without private residual, Mirror phase with/without private residual, and an ordinary scalar phase with the same implementation.

Fresh client descriptors were supplied to both methods; unseen clients had no local examples. Payloads were deterministic ZIP/NPY FP32 archives reloaded before evaluation. Dev seeds 34101/34102 were used. The Mirror phase payload was 11.7% and 11.2% smaller than pFedHN with private fallback, below the frozen 20% threshold. The native scalar-phase control was then set to use exactly the same inference arrays and metadata as Mirror; its payload hashes matched exactly. Fresh seeds 34111–34113 remain sealed.

No optimizer updates were run. Each world used 1,536 client support examples plus 32 off-orbit support examples, and 32,768 held-out client examples. The generator fit was least squares. The report is a mechanism/storage screen, not a complete federated training comparison.

## D — Decision

**FAIL at development for the frozen storage margin and Mirror-specific value.** Unseen-client mean nMSE was below 5e-15 for both pFedHN and Mirror phase in both worlds. With private off-orbit state, pFedHN used 1,721–1,730B and Mirror used 1,527–1,528B, only 11.2–11.7% less than the frozen 20% gate. The native scalar-phase control had exactly the same serialized byte count and hash as Mirror in both worlds. The no-private phase model had off-orbit nMSE 0.60–0.70; the paid residual restored near-zero error. Independent 65-client state used about 20.1KB, while one shared FedAvg matrix had worse unseen-client quality (nMSE 0.055–0.115).

Mirror phase reduced per-client code communication from 8B to 4B, but the server generator dominates total bytes. This is a narrow communication-state reduction, not a Mirror-specific gain under the controls.

## Fact / Interpretation / Hypothesis

**Fact:** All 16 unseen clients were predicted at near-zero nMSE. Mirror payload was 1,527/1,528B vs pFedHN 1,730/1,721B. The ordinary scalar-phase control matched Mirror byte-for-byte. Development payload/metric replay is exact.

**Interpretation:** A scalar code can halve client metadata relative to a 2D embedding for this planted circular client family, but the total storage margin is modest and the same result follows from ordinary polar coordinates.

**Hypothesis:** Larger client banks or communication-constrained deployments may benefit from phase-like coordinates, but this experiment does not establish adoption or Mirror-specific quality/storage gains.

## C — Strongest counter-hypothesis

The client descriptors lie on a known unit circle, so replacing `(cos θ, sin θ)` with `θ` is a generic coordinate change. The shared linear generator and planted task manifold explain the quality and compression.

## U — Unconfirmed

No neural hypernetwork training, heterogeneous real client data, communication rounds, privacy constraints, client drift, unknown descriptors, or fresh-world replication was tested. The pFedHN control is a linear full-matrix generator, not a reproduction of the original nonlinear training stack.

## Reproduction

```bash
python experiments/mirror_applications/ma-341-federated-mirror-vs-pfedhn/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-341-federated-mirror-vs-pfedhn/tests
python experiments/mirror_applications/ma-341-federated-mirror-vs-pfedhn/source/verify.py
```
