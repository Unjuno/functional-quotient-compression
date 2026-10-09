# MA-533 — Trained transcoder decoder atoms as feature views

Status: **FAIL**. Branch: `research/ma-533-transcoder-feature-bank-20261009`. Frozen protocol hash: `29208e278d9d5573fe17979b0125ca6ace6d689e1d2afe9243d41d4c63d12671`. Prior art: PA103 (transcoders) and PA102 (SAE steering).

## H — Hypothesis

A support-trained sparse transcoder decoder dictionary can provide a better atom source than the pretrained SAE bank: one shared 16-row decoder basis plus per-task coefficients should preserve explicit-FV behavior within .10 gold-logprob nats/.05 accuracy and use no more than .99x its actual serialized bytes.

## T — Frozen execution

Pinned Pythia-70m. For each seed, tasks 0–11 support examples supplied layer-3 post-attention-normalization inputs and matching MLP outputs. A 512→2048 ReLU top-32 encoder and 2048→512 decoder trained for exactly 1,000 Adam updates, batch 128, MSE + .001 mean-absolute latent penalty. Tasks 12–15 and all evaluation queries were excluded from transcoder fitting. A shared 16-atom decoder pool was chosen by fit-task FV residual correlation, and all 16 task FVs were coded with OMP16. Controls: no intervention, explicit FV, global transcoder OMP16 (full decoder charged), and global SAE OMP16 (pinned checkpoint charged once).

The inference candidate stores only selected decoder rows, pool IDs and task code. It does not need the transcoder encoder or optimizer state. Both code and dictionary bytes are included in the uncompressed NPZ. The SAE control pays the complete pinned checkpoint separately from its code-only NPZ. The first development attempt double-counted the same SAE dictionary in two locations; exact pre-amendment metrics and payload hashes are retained in `pre_amendment_1/`. Both seeds were rerun after correcting accounting; the protocol hash and quality metrics did not change.

## D — Decision

**FAIL; fresh data stays sealed.** The transcoder reconstructs its fit-support MLP outputs well (FVU .0159/.0157), but its selected atoms do not represent the held-out task FVs well, and its selected-basis payload exceeds explicit FVs.

| Seed | Method | Payload B | Accuracy | Gold log-prob | Δ vs explicit FV |
|---|---|---:|---:|---:|---:|
| 53301 | Explicit FV | 34,214 | .3125 | −10.292674 | — |
| 53301 | Shared transcoder pool16 | 36,294 | .2188 | −12.692569 | −2.399894 |
| 53301 | Global transcoder OMP16 | 4,197,810 | .2500 | −12.414637 | −2.121963 |
| 53301 | Global SAE OMP16 | 3,840 code + 4,204,391 SAE | .2188 | −11.070990 | −.778315 |
| 53302 | Explicit FV | 34,214 | .2188 | −10.739561 | — |
| 53302 | Shared transcoder pool16 | 36,294 | .2188 | −13.028971 | −2.289410 |
| 53302 | Global transcoder OMP16 | 4,197,810 | .2188 | −12.378145 | −1.638584 |
| 53302 | Global SAE OMP16 | 3,840 code + 4,204,391 SAE | .1875 | −11.458640 | −.719079 |

Shared transcoder payload is 2,080 B larger than the explicit-FV bank (1.061x). Standalone Pythia+shared transcoder state is likewise 2,080 B larger than Pythia+explicit FVs. Global transcoder OMP is 4,163,596 B larger standalone. Global SAE OMP is 4,174,017 B larger standalone. The trained dictionary reduces its own MLP-output reconstruction error but does not transfer that reconstruction quality to task-FV interventions.

Support-only activation capture used 6,371 tokens and 96 prompts in seed 53301; exact seed-53302 counts/timing are retained in metrics. Transcoder fit took 18.11/20.51 CPU seconds at the frozen 805,306,368,000 operation proxy. Shared pool selection and code construction took .0156/.0352 s and .0040/.0040 s. These CPU measurements are not optimized runtime claims. Fresh seeds 53311–53313 remain locked.

## Evidence classes

**Facts:** Four tests pass. Deterministic replay matched eight paid payload hashes, core metrics, support/evaluation splits, transcoder decoder and selected pool. MLP FVU and feature sparsity are stored per seed. Fresh data was not accessed.

**Interpretation:** A good fit to the MLP's own output task does not imply that decoder atoms span the support-derived relation FVs. The selected atom source changes, but the quality/byte frontier remains worse than explicit FVs, while global dictionary controls cost several megabytes.

**Hypothesis:** Transcoders trained directly on intervention targets or substantially wider shared bases may behave differently. Those require a new target/task design and private-residual frontier.

## C — Strongest counter-hypothesis

The failure may be target mismatch: this transcoder models the layer-3 MLP's ordinary forward output, while the benchmark asks it to span support-derived cross-task activation differences. The low MLP FVU does not validate intervention atoms.

## U — Boundaries

One small Pythia model, one support-trained transcoder architecture and four held-out relation tasks. No broad transcoder steering, near-convergence capacity, fresh-world or natural MoE claim is established.
