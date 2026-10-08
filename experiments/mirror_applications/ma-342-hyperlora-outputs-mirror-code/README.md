# MA-342 — HyperLoRA full factors vs Mirror functional code

Status: **FAIL at the development gate for Mirror-specific storage value; fresh seeds sealed**  
Dedicated branch: `research/ma-342-hyperlora-outputs-mirror-code-20261008`  
Prior art: PA38 HyperLoRA federated personalization.

## H — Hypothesis

A shared rank-2 LoRA basis plus a scalar client phase may replace a HyperLoRA generator that emits complete client A/B factors, generalize to clients without local data, and reduce actual bytes. Direct coefficient and ordinary scalar phase controls test whether any result is Mirror-specific. A client outside the orbit may require private factor state.

## T — Test

Synthetic 16×16 linear client models. Sixty-four clients lie on an orbit with shared rank-2 left LoRA factor and sinusoidal right-factor views; 48 clients have 40 local support examples and 16 are held out from fitting. One off-orbit client has 40 local examples and a private right factor. Methods: independent rank-2 adapters, one averaged adapter, a linear HyperLoRA-style generator emitting full A/B factors from 2D client embeddings, the generator with/without private state, Mirror phase with/without private state, direct two-coefficient basis codes, and native scalar phase. The shared subspace and generator are fitted by least squares to support-derived client matrices. Evaluation uses 512 independent inputs per client.

The two development worlds were 34201/34202. Fresh seeds 34211–34213 were not accessed after the frozen development stop. Deterministic ZIP/NPY FP32 payloads were reloaded before inference. No optimizer updates were performed.

## D — Decision

**FAIL at development for the frozen storage gate.** Both Mirror and HyperLoRA predicted the 16 unseen aligned clients at nMSE below 7e-15. With private off-orbit fallback, HyperLoRA payload was 2,861B/2,874B while Mirror was 2,980B/2,976B; the Mirror payload was larger in both worlds and therefore missed the required 20% reduction. Direct coefficients used 3,157B/3,170B. The ordinary scalar phase control exactly matched Mirror payload hashes and quality. Mirror reduced client-code communication from 8B to 4B and its reported factor-view MAC proxy was lower, but those changes did not improve the serialized storage point. Without private state, off-orbit nMSE was 0.42–0.60; a private right factor restored near-zero error.

Independent 65-client LoRA factors used about 39.8KB. One averaged adapter had unseen-client nMSE 0.17–0.26.

## Fact / Interpretation / Hypothesis

**Fact:** Mirror missed the pFedHN-style HyperLoRA byte gate on both development seeds; the scalar phase control was byte/hash identical. Fresh seeds remained sealed.

**Interpretation:** The task family is representable by both the phase view and the full-factor generator. ZIP compression favored the generator's sparse structured factor-output weights, offsetting the apparent one-scalar code advantage. An off-orbit client crosses the private-state boundary.

**Hypothesis:** A learned/nonlinear client distribution or a larger bank may change the storage/compute tradeoff, but requires a new protocol with realistic HyperLoRA training and client heterogeneity.

## C — Strongest counter-hypothesis

The planted client orbit is exactly sinusoidal and the linear HyperLoRA output map already captures it. Its structured zeros compress well in the actual payload, while the phase code adds no unique function.

## U — Unconfirmed

No nonlinear federated training, local optimizer steps, privacy/communication rounds, client drift, unknown client features, or fresh-world replication. HyperLoRA is represented by a linear full-factor generator, not the full paper's learned training stack.

## Reproduction

```bash
python experiments/mirror_applications/ma-342-hyperlora-outputs-mirror-code/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-342-hyperlora-outputs-mirror-code/tests
python experiments/mirror_applications/ma-342-hyperlora-outputs-mirror-code/source/verify.py
```
