# MA-591 results — soft prompt bank compression

## Fact

Two registered seeds each trained eight independent 8-token soft prompts on separate WikiText articles using frozen Pythia-70M. Training used 64 optimizer updates, four 64-token windows per update, and 16,384 train tokens per seed. Each task was evaluated on four 64-context/64-target windows from article token suffix 1024:1536, disjoint from prompt fitting. Amendment 1 added the preregistered inference-time metric; initial runs were retained but superseded. A replay of amended seed 59101 reproduced all NLLs, task titles and serialized byte counts exactly.

| Method | NLL, seed 59101 / 59102 | Delta vs full prompt | Serialized prompt bank bytes | Mean inference sec/task, 59101 / 59102 |
|---|---:|---:|---:|---:|
| Zero prompt | 4.63199 / 4.28355 | +0.16562 / +0.13306 nat | 67,556 / 68,068 B | 0.320 / 0.302 |
| Full independent prompts | 4.46637 / 4.15049 | 0 / 0 nat | 67,556 / 68,068 B | 0.303 / 0.283 |
| Shared rank-4 Mirror | 4.49094 / 4.16399 | +0.02456 / +0.01350 nat | 43,526 / 44,038 B | 0.280 / 0.295 |
| Native shared rank-4 | 4.49094 / 4.16399 | +0.02456 / +0.01350 nat | 43,526 / 44,038 B | 0.285 / 0.279 |

The shared rank-4 bank reduces serialized bytes by 35.6% in both seeds and stays within the +0.05 nat/token full-prompt quality limit. It improves over zero prompts by 0.141/0.120 nat/token. Mirror and native bank archives are byte-identical for each seed. The shared state contains a 4096-value FP16 mean, 4096×4 FP16 basis and 8×4 FP16 coordinates plus metadata; independent state stores eight 8×512 FP16 prompts plus task metadata. All are charged by actual NPZ bytes.

Training took 26.60/25.46 s total (64 optimizer steps, 3.33/3.18 s per task). Rank-4 prompt reconstruction took 0.115/0.126 ms per task. Evaluation time across four windows averaged 0.280/0.295 s per task for Mirror, 0.303/0.283 s for full prompts. Every deployed representation adds eight virtual input tokens; measured added embedding operation proxy is 4096 per example. These CPU timings are unoptimized.

## Interpretation

**D: FAIL for Mirror-specific attribution.** The low-rank bank passes the frozen quality and storage gates and preserves prompt utility over zero prompts. However, the exact serialized codebook is ordinary native rank-4 PCA/low-rank prompt coding, so there is no incremental Mirror mechanism. Fresh tasks remain sealed because this exact native alias is already decisive.

## H / T / D / C / U

**H:** Eight independent task prompts can be represented by a shared rank-4 basis and tiny per-task coordinates at lower bytes without losing held-out continuation quality.

**T:** Frozen Pythia-70M, two seeds, eight WikiText article tasks per seed, 64 prompt-tuning updates per bank, rank-4 PCA compression, zero/full/native controls, held-out article suffix NLL, actual serialized bytes, optimizer/runtime and virtual-token costs. Amended registered runs and one exact same-seed replay are recorded in `ARTIFACT_PROVENANCE.json`.

**D:** FAIL for Mirror-specific value. Quality and storage gates pass in both dev banks, but shared rank-4 prompts exactly alias native low-rank coding. Fresh remains sealed.

**C:** The result is ordinary low-rank prompt-bank compression; no evidence distinguishes Mirror coordinates from standard shared-basis factorization.

**U:** Generalization to new articles/tasks, more task banks, other ranks, downstream task accuracy beyond next-token NLL, and accelerator inference.

## Fact / Interpretation / Hypothesis

- **Fact:** 35.6% fewer serialized prompt-bank bytes with NLL deltas +0.0246/+0.0135 nat; exact native-control equality.
- **Interpretation:** A useful prompt storage frontier exists in this narrow screen, but its gain belongs to standard low-rank coding rather than Mirror-specific functional freedom.
- **Hypothesis:** Other structured View families might beat native low-rank coding, but require distinct preregistered controls.
