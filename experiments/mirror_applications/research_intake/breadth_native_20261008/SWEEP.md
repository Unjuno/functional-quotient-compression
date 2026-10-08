# 2026-10-08 — Broad literature-to-Mirror method intake (no priority override)

**Branch:** `research/mirror-breadth-method-sweep-20261008` — research only. **User correction:** previously singled-out MA-1175 no longer holds first rank. The central MA priority status tags P0/P1 and active worker's MA-255 are unchanged.

## Why these instead of another same-idea variation

The prior 1176-row registry contains many MoE, KV, low-rank, encoder and model-composition roles, but no explicitly named UniSparse, LoGo, RanLoRA, efficient ESM protein property bank, CellFM/RegFormer gene perturbation, or ProtoMech protein circuits. Each new idea targets a specific *native physical object* that has been shown useful; no claims are transferred from the native method to Mirror.

## Six distinct new UNTESTED plans

| ID | Minimal marginal m experiment | Mandatory hardest baseline | Interest class |
|---|---|---|---|
| [MA-1177](plans/MA-1177/README.md) | Role m on native composite-token sparse attention readout | UniSparse and same-byte linear role head | P0 |
| [MA-1178](plans/MA-1178/README.md) | Smaller physical adapter library beneath LoGo instance merger | Original LoGo, SVD-compressed LoGo, ALoRA | P0 |
| [MA-1179](plans/MA-1179/README.md) | Short task code in residual nonlinear RanLoRA | Original RanLoRA Hadamard modulation, ordinary task core | P1 |
| [MA-1180](plans/MA-1180/README.md) | Multiple protein assay/property outputs from one ESME/ESM2 encoder | Native ESME and native one-trunk independent heads | P1 |
| [MA-1181](plans/MA-1181/README.md) | Factorized gene1×gene2×cell perturbation readout | Native CellFM/RegFormer, GEARS, calibrated mean/linear | P1 |
| [MA-1182](plans/MA-1182/README.md) | Protein function circuit roles from original ProtoMech CLT | Native CLT/windowed CLT/PLT, ordinary sparse linear code | P1 |

Each folder contains a complete H/T/D/C/U READ ME, tensor shapes, native method, exact insertion, CPU mechanism fixture, source/dev/fresh firewall, name of hardest controls, adoption boundary and machine-readable `PROTOCOL.json` + `STATUS.md`. These are not active experiments. Full benchmark reproduction requires the native author implementation, model revision and dataset license/hash.

## Primary-paper verification

- **PA440** — UniSparse multi-granularity sparse composite-token construction: https://proceedings.mlr.press/v306/liu26h.html
- **PA441** — The Sparse Frontier attention kernel/selection-method controls: https://aclanthology.org/2026.findings-acl.1926/
- **PA442** — LoGo instance-level training-free adapter selection and merging: https://aclanthology.org/2026.acl-long.1837/
- **PA443** — RanLoRA native nonlinear residual and Hadamard modulation: https://aclanthology.org/2026.findings-acl.852/
- **PA444** — ESME efficient protein language model and PEFT heads: https://doi.org/10.1016/j.isci.2025.113495
- **PA445** — ProtoMech native cross-layer protein circuits: https://proceedings.mlr.press/v306/tsui26a.html
- **PA446** — CellFM native perturbed-cell embedding/LoRA: https://www.nature.com/articles/s41467-025-59926-5
- **PA447** — RegFormer native GRN-guided Mamba embedding: https://www.nature.com/articles/s41467-026-72198-x
- **PA448** — Nature Methods: strong simple perturbation model baselines: https://www.nature.com/articles/s41592-025-02772-6
- **PA449** — Nature Biotechnology: metric calibration / positive control: https://www.nature.com/articles/s41587-026-03307-w
- **PA450** — ExpertFlow native routing prediction and cache: https://doi.org/10.1145/3770743.3804292
- **PA451** — DyMoE mixed precision and lookahead native control: https://arxiv.org/abs/2603.19172
- **PA452** — Single-cell foundation model cross-method utility benchmark: https://doi.org/10.1002/advs.202514490

Biological perturbation metrics are an especially important falsification case: PA448 found simple/linear models competitive on studied tasks, while PA449 showed that proper positive-control metric calibration can change apparent deep-model performance. Require both controls; do not equate foundation-model parameter count with predictive utility. CellFM published model source also includes an authors' correction note, so re-audit exact corrected native comparison.

## Existing hypotheses: strengthen controls instead of duplicate ID

- [MA-1176 — ExpertFlow / DyMoE / MoE-Infinity direct serving controls](existing/MA-1176-EXPERTFLOW_DYMOE_NATIVE.md): PA450, PA451, PA434, PA435, PA436, PA439
- [MA-1172 — UniSparse/Sparse Frontier coupling-aware cache allocator controls](existing/MA-1172-UNISPARSE_SELECTOR_CONTROL.md): PA440, PA441, PA425, PA426
- [MA-1114 — LoGo training-free instance merging as strong adapter-count break-even](existing/MA-1114-LOGO_ADAPTER_AMORTIZATION.md): PA442, PA427, PA351, PA352
- [MA-1099 — RanLoRA native nonlinear Hadamard controls for structured core](existing/MA-1099-RANLORA_NONLINEAR_CONTROL.md): PA443, PA427, PA373
- [MA-1175 — No special priority + biological multi-output stress controls](existing/MA-1175-NEUTRAL_MULTI_OUTPUT_AND_BIO.md): PA42, PA437, PA438, PA444, PA445

## Worker-free implementation sequence

1. Select an existing registered candidate based on available models/data/hardware and expected information gain, NOT recent discussion order or branch sorting.
2. Use its local README to implement native algebra / small mechanism. For a real-paper reproduction, also check published source/commit and checkpoints. Log any source unavailability as BLOCKED.
3. Before opening a new fresh set, freeze native/candidate model code hashes, dataset/preprocessing lineage, 3 dev world seeds and 5 fresh disjoint task-world identities. Never refit after viewing fresh.
4. Run exact shape/unit/gauge/control tests; verify that a new m changes *useful task output* rather than equivalent gauge.
5. Compare strongest native + cheap equal-byte diagonal/linear/group factor vs Mirror. Paid bytes include shared basis, native tokenizer/embedding, selection indices, router, codes, buffers and optional private exception. K output branches may share trunk compute, but do not count native one-trunk multi-head as K independent heavy forwards.
6. Report pairwise heldout quality/worst-task, serialized file sizes, active compute and real CPU/GPU P50/P95. If CUDA absent, no on-device prefetch or block-sparse speedup claim. Report all negative results.
7. Five-world PASS is a *mechanism screen*. For ADOPTED require independent >=10-world replication, natural task, native-paper functionality and strict marginal Pareto gain.

## Program safety

The latest **canonical worker** branch remains independently `research/mirror-application-worker-ready-20261007` with MA-255 next. The new entries are staged only on `research/mirror-breadth-method-sweep-20261008`. The prior studies' existing 29 PROMISING/18 FAIL statuses, main, WORKER_QUEUE, WORKER_START_HERE, CONTEXT_ROUTER, first queue and active experiment files remain unchanged. Check ID collisions again before any manual promotion.
