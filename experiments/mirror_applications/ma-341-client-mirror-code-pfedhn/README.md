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
