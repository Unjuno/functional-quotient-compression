# MA-341 — Client Mirror code vs pFedHN full-model generation

Status: SCREENING. Prior art: PA40 pFedHN.

## H

For client tasks whose linear predictors lie on a shared circular two-direction orbit, a client phase `m` plus shared matrices may generalize to held-out client identities with fewer actual bytes than a pFedHN-style model generator. The Mirror-specific byte margin may disappear against an ordinary two-coefficient shared-basis control.

## Mirror insertion

> **Mirror insertion:** this experiment adds one client phase `m` to a shared base predictor and two shared task directions, producing client-specific linear predictors without storing a full predictor per client.

Compare a single shared predictor, phase-coded shared basis, ordinary two-coefficient shared basis, pFedHN-style MLP hypernetwork that generates full predictor weights from client descriptors, and supervised independent client predictors. Client descriptors and all generator/basis state are paid. pFedHN receives the same 2D client descriptor as the shared methods.

## T

Input dimension 8, output dimension 4, 16 seen training clients and 8 held-out client phases interpolated between train phases. Each client has labeled support examples; held-out test examples are disjoint. Teacher matrices are generated as W(phi)=W0+cos(phi)A+sin(phi)B with small observation noise. Train Mirror/basis controls and pFedHN from seen client data only. Development worlds 34121–34122, fresh worlds 34131–34133. Report held-out MSE, payload bytes, optimizer updates, training time, inference MAC proxy and per-client communication bytes.

## Gates

**PROMISING:** Mirror and pFedHN meet held-out MSE <=1e-3; Mirror saves >=20% actual bytes over pFedHN and >=10% over ordinary coefficient control at comparable quality. **FAIL:** a simpler coefficient control matches Mirror within 5% bytes/quality, or Mirror misses quality. Independent per-client regression is an upper reference; its extra support is reported.

## C / U

The teacher is intentionally aligned to a two-direction circular orbit. Results cannot establish natural federated personalization. No client privacy, non-IID distribution shift, communication protocol or real federated hardware is tested.


## Results

**H:** phase-coded shared basis may generalize to held-out client phases more compactly than a pFedHN-style model generator; a direct coefficient control determines whether the gain is Mirror-specific.

**T:** three fresh worlds, 16 seen clients × 64 training examples and 8 held-out midpoint clients × 256 test examples. Each task is a linear predictor on a planted circular rank-2 orbit. Mirror, direct coefficient basis, pFedHN-style 2→32→32 generator and single shared predictor each received 1,000 Adam updates; the independent reference used held-out support data by least squares. Client phase/descriptor bytes are included.

**D — FAIL for Mirror-specific gain:** fresh mean held-out MSE: Mirror 1.016e-4 at 516 B; generic two-coefficient basis 1.016e-4 at 556 B; pFedHN-style generator 1.184e-4 at 4,783 B; shared single predictor .2696 at 281 B; independent support-fit 1.130e-4 at 1,124 B. Mirror uses 89.2% fewer payload bytes than this pFedHN-style generator, but only 7.2% fewer bytes than the functionally identical coefficient control, missing the preregistered 10% threshold. Mean training wall time was .371 s Mirror, .351 s generic coefficients, .415 s hypernetwork.

**C:** all tasks are deliberately planted on the same circular rank-2 orbit, which favors the phase and coefficient basis. The hypernetwork is a small synthetic pFedHN-style MLP, not a reproduced federated system.

**U:** natural client heterogeneity, local training/communication rounds, privacy, robustness, distribution shift and real federated datasets are untested.

### Fact / interpretation / hypothesis

**Fact:** 15 fresh rows across three worlds. Metric and payload hash replay difference is zero; two tests pass.

**Interpretation:** the structured client code is compact versus full model generation, but its marginal advantage over direct coefficients is too small to call Mirror-specific under the fixed gate.

**Hypothesis:** on natural clients, private residuals or a native conditional hypernetwork may erase the phase basis advantage; this synthetic orbit provides no evidence either way.
