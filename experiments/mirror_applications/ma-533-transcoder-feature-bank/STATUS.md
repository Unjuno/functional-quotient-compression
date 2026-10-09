# MA-533 status

- Status: **FAIL**
- Branch: `research/ma-533-transcoder-feature-bank-20261009`
- Protocol froze before implementation; accounting amendment did not change gates or protocol.
- Both development seeds rerun after correcting duplicated SAE dictionary charge; fresh 53311–53313 remain sealed.
- Four tests pass; eight paid payload hashes, core metrics, split manifests, transcoder decoders and selected pools replay exactly.
- Pre-amendment metrics and payload SHA-256 manifests are retained under `pre_amendment_1/`.

## H / T / D / C / U

- **H:** task-trained transcoder decoder atoms form a compact shared basis for relation-task FVs.
- **T:** support-only fit tasks0–11 train a 2048-feature top-32 layer3 MLP transcoder; decoder pool16 chosen from fit FVs; explicit FV, global transcoder OMP16 and global SAE OMP16 controls.
- **D:** FAIL. Shared code loses 2.400/2.289 nats and .094/0 accuracy to explicit FV; payload 36,294 B vs 34,214 B. Train MLP FVU is good (.0159/.0157), but global SAE sparse coding is more accurate and uses smaller incremental codes when SAE is loaded.
- **C:** atom target mismatch: ordinary MLP output reconstruction differs from the task-FV intervention target.
- **U:** other transcoder objectives, natural feature steering, fresh seeds, wider bases/private residuals and near-convergence remain untested.

## Evidence classes

- **Facts:** maximum core-metric replay difference 0; eight final payload hashes exact; model/SAE hashes and support/evaluation splits checked; fresh unopened.
- **Interpretation:** ordinary-MLP transcoder quality does not transfer to this FV task; no quality/storage Pareto gain was established.
- **Hypothesis:** transcoder features trained directly on intervention targets may work; this is a distinct future target design.
