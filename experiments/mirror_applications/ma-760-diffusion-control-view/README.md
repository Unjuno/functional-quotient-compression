# MA-760 — diffusion control View + private residual

Status: **FAIL (frozen development gate)**
Branch: `research/ma-760-diffusion-control-view-20261008`
Baseline: `407ca7e2047326d1e4b753e55e05c4730f26f32b`
Draw 6: uniform draw from 529 eligible rows; seed `68531eb795d3e4a5bc9bbfceddff040760474c10ce19d0fef5bb034e14f54cdd`, zero-based index 259. Full provenance and pool snapshot: `source/random_draw.json`, `source/selection_pool.csv`.

## Hypothesis

A shared control-adapter basis with condition-specific Mirror coordinates and a sparse private residual can preserve held-out spatial-control behavior while reducing per-condition state versus independent branches. The private-state requirement should grow with control heterogeneity.

## Prior art and scope

PA190 (ControlNet) uses copied condition-specific branches; PA191 (T2I-Adapter) is the low-cost independent adapter control. This experiment tests the shared/private boundary described by MA-760.

The environment has CPU only. We loaded the pinned `hf-internal-testing/tiny-stable-diffusion-pipe` UNet as an integration substrate and froze it. This internal test fixture is not a quality pretrained image generator. Conditions and target residuals are synthetic 3-channel maps and generated 4×3×3×3 convolution operators. No real edge/depth/pose pairs, image-level generation metric, or ControlNet/T2I-Adapter reproduction was run.

## T — protocol and execution

- Development seeds 7601/7602; heterogeneity `rho ∈ {0.1, 0.4, 1.0}`; six basis-training condition identities and two held-out identities per seed.
- Compared no control, hard shared adapter, independent full per-condition 3×3 adapters, shared additive low-rank basis, multiplicative Mirror coordinates over that same basis, and Mirror plus top-8 sparse private kernel coefficients. Ranks `{1,2,4,8}`.
- Direct condition kernels and held-out addresses were solved from declared synthetic calibration examples with float64 ridge/least squares. **Zero optimizer updates**; this was frozen in the protocol before scoring.
- 180 held-out evaluation rows. Total wall time 5.28s on CPU. Frozen UNet batch-2 forward calibration: 7.17ms. Actual U-Net output tensors were used to check adapter composition; primary metric is synthetic held-out residual MSE.
- Each K=8 library artifact and each held-out condition state was serialized and hash checked. Whole-library bytes add the actual serialized 5,830,450-byte UNet payload to the adapter library. The shared UNet dominates absolute storage.

## D — FAIL

No rank passed the full conjunction on both seeds, both held-out identities, and all heterogeneity levels. Fresh seeds/conditions remain unopened.

### Fact

Mean normalized residual MSE across two seeds and two held-out identities:

| rho | Hard shared | Mirror rank 4 | Mirror + private rank 4 | Independent full |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0185 | 0.0078 | 0.0053 | ~0 |
| 0.4 | 0.2304 | 0.0969 | 0.0632 | ~0 |
| 1.0 | 0.7420 | 0.3133 | 0.1950 | ~0 |

The direct independent adapter fits this noiseless linear target family to numerical zero. The private residual lowers Mirror rank-4 error, but its condition payload is 2,305 bytes versus 2,020 bytes for an independent adapter. It therefore does not reduce the marginal state in this small fixture.

At rank 4, Mirror-only stored 4,775 adapter-library bytes versus 5,406 for independent: 631 bytes saved in the controller bank. Once the shared UNet is charged, complete payloads were 5,835,225 versus 5,835,856 bytes, a **0.0108%** reduction. Rank 8 exceeded independent whole-library bytes. Mirror + private rank 4 used 5,836,611 total bytes, 755 bytes more than independent.

Multiplicative Mirror and additive low-rank produced exactly equal held-out MSE at every tested rank/rho/seed/condition. The preregistered Mirror-specific requirement was ≥5% over the byte-near additive control; it failed everywhere. Fresh was not opened.

### Interpretation

The synthetic screen maps a clear quality/private-state tradeoff: hard tying degrades as condition operators diverge; shared coordinates recover part of the error; private coefficients recover more at added bytes. In this setup, the Mirror coordinate is only a low-rank coefficient address and adds no function beyond the additive basis control. The tiny fixed UNet makes a large controller-bank percentage reduction invisible in total model storage.

### H / T / D / C / U

- **H:** A shared condition basis plus Mirror address and sparse private exceptions can preserve control behavior with less state, with the private break point increasing as heterogeneity rises.
- **T:** Tiny frozen UNet; synthetic spatial residual operators; two development seeds; three heterogeneity levels; six basis conditions and two held-out conditions; 180 scored rows; no optimizer updates; exact serialized payload bytes.
- **D:** **FAIL.** Independent adapters were nearly exact; private Mirror did not meet the quality gate or per-condition byte gate; Mirror tied additive low-rank and failed the ≥5% Mirror-specific clause. Fresh stayed unopened.
- **C:** The screen is controlled by a noiseless linear teacher that favors direct independent kernels, and its tiny shared UNet dominates whole-model bytes. This may not predict nonlinear ControlNet branches on real paired data.
- **U:** Real edge/depth/pose pairs, a quality-pretrained UNet, full ControlNet/T2I-Adapter architectures, image-level generation quality, GPU memory/FPS/energy, and fresh conditions remain untested.

## Protocol amendment before development

Before any development scores, the initial 5% whole-payload reduction threshold was amended to require a strict actual-byte reduction plus exact byte and percentage reporting. The shared tiny UNet dominates this toy adapter payload, so a 5% total-bundle reduction was unreachable by construction. Quality and Mirror-specific thresholds were not relaxed. Details are recorded in `PROTOCOL.json`.

**BOUNDARY:** mechanism screen on synthetic spatial residuals; no claim about real diffusion control quality or production ControlNet compression.
