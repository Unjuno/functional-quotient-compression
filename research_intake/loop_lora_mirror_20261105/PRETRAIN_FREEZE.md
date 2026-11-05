# LOOP-LORA-MIRROR-20261105 — pre-training freeze

This is a separate standalone research run, not a redefinition of MA-1184 or a registered new MA ID. The user requested actual Loop/LoRA/Mirror experiments. Historical results and main remain unchanged.

## Frozen local evidence
- Local source commit: `cacab4371e61bfb508a92c53ad186fa8a62712b8`.
- Complete PROTOCOL.json SHA256: `8a32f413a67cb3eb1099f23c1d0596a0c6b65aff633421926148b6e7070f3260`.
- SOURCE_FREEZE.json contains hashes of 10 protocol, plan, source and test files. They are checked before main training and evaluation. A complete source/data/checkpoint archive will be supplied with the results, not implied to exist in this directory.
- Four pilot runs only have completed. Main comparison and final evaluation have NOT started at the time of this freeze.
- Fourteen unit tests passed, including common initial functions, double-reflection inverse, code gradients, disjoint graph identities, exact packet reload, LoRA factor-gauge invariance and frozen-backbone protection.

## Task and model
Three-hop successor prediction on eight-node graphs. TRAIN/DEV/final evaluation graph identities are disjoint single directed eight-cycles; topology OOD uses two disjoint four-cycles. Node vocabulary and number of records are unchanged. One full-attention, pre-LayerNorm Transformer block with d=32, 2 heads, ReLU FFN width48, repeated R=1,2,4,8. Unordered edge tokens plus query, no positions, no RoPE, no causal mask and no MoE. This is an algorithmic Transformer experiment, not a causal LM benchmark or reproduction of an LLM paper.

## Stage A: joint training
Five paired seeds: 21001,21011,21019,21023,21031. Six methods: shared, FiLM, LoRA rank1, LoRA rank2, paired-Householder Mirror, Mirror+LoRA rank1. Total 120 training runs. Identical base initialization and minibatch stream within seed and R. FiLM and LoRA start as identity updates. Mirror uses two initially identical reflection directions, hence identity as a function up to roundoff. Mirror is before/after the FFN nonlinearity; LoRA modifies both FFN projections. All common parameters and codes train jointly.

Each run: 6000 train queries, 512 DEV, 800 AdamW updates, batch64, lr0.003, weight_decay0.0001, clip1. Checkpoints selected by DEV CE every100 updates including step0. The independent pilot seed19001 tried lr0.0015/.003 on shared/LoRA2 R4: at .003 DEV accuracy was 61.52%/65.82%, avoiding saturation. No Mirror result was used to choose the global learning rate.

## Stage B: replacement of learned LoRA
All ten R4/R8 LoRA2 checkpoints, without cherry-picking, serve as teachers. Four replacements: FiLM, LoRA1, Mirror, Mirror+LoRA1. Total40 additional fits. Copy and freeze every shared teacher parameter; fit only replacement codes to the teacher's FFN outputs at the teacher's TRAIN trajectories (1024 queries). DEV traces (256 queries) select checkpoints. 600 AdamW updates, batch64, lr0.003, weight_decay0. Local FFN squared error is normalized by the TRAIN error after deleting teacher adapters. No target-task labels train replacement codes.

After all 160 checkpoints are hashed and sealed, instantiate final 1024 fresh queries and1024 topology-OOD queries per seed. Evaluate both teacher-forced local FFN error recovery AND autonomous full-rollout accuracy, CE, KL and teacher agreement. No teacher forcing in full-rollout scoring.

## Frozen decisions and accounting
- Primary native Mirror screen: at R4, Fresh CE at least10% lower than both FiLM and LoRA1 in >=4/5 paired seeds. Exploratory screen, not a universal/capacity claim.
- Replacement mechanism: teacher accuracy exceeds adapter-deletion accuracy by >=5 percentage points, local FFN error recovery >=80%, and full-rollout accuracy no more than2 points below teacher, in >=4/5 R4 seeds. If the teacher does not need LoRA, do not claim useful compression.
- Hybrid secondary: R4 Mirror+LoRA1 accuracy within2 points of LoRA2 in >=4/5; code-count saving is separate from statistical noninferiority or runtime advantage.
- Actual total inference bytes, adapter counts, batch1/batch64 CPU timing and arithmetic proxies are reported. Paired update counts are NOT matched FLOPs, especially across loop counts. StageB costs include the trained teacher and fitting.
- No efficiency/adoption claim unless quality, storage and runtime support it together. Preserve all failures. No additional hypothesis is substituted after final scores.

## Environment and verification
AMD EPYC9V74, observed2.596GHz not fixed; Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, FP32 eager, no GPU. Four workers, one numerical thread each. Explicit attention matmul/softmax, no FlashAttention. All checkpoint hashes and full-rollout scores must replay; all frozen-backbone tensors must remain unchanged. Two deterministic retraining checks are planned.

Prior art: Bae et al., Relaxed Recursive Transformers, arXiv:2410.20672v3. Inspiration only; their LLM results are not claimed here.
