# MA-1185 candidate — Loop-specific LoRA replacement by nonlinear Mirror

2026-11-04. New branch scoped run; central registry last observed maximum MA-1184. This candidate is not a completed/confirmed finding and this file does not change prior verdicts.

## Objective
Test whether the corrections learned by loop-specific FFN LoRA can be replaced by smaller Householder view codes, with real images rather than a Mirror-generated synthetic teacher. This is a small recurrent visual Transformer diagnostic, NOT a reproduction of the LLM experiments in Relaxed Recursive Transformers (arXiv:2410.20672).

## Frozen design
- Data: sklearn bundled handwritten digits, 1797 images, fixed stratified 1078 TRAIN / 359 DEV / 360 TEST. Split seeds 314159 and 271828. Known-range pixel normalization /16. Four 4x4 patches plus class token. TEST assigned and hash recorded but not evaluated until all checkpoints are sealed.
- Model: shared Transformer block, model dimension32, FFN64/GELU, 4 attention heads, learned absolute position embeddings; source trained with2 loops. Source, head, norms and attention frozen for extension to4 and8 loops.
- Five initialization seeds:11,23,37,53,71. Same dataset split across these seeds; NOT independent datasets.
- Source600 AdamW updates, lr.003. Adapter training300 updates; distillation400 updates. Batch64, weight decay1e-4, clipping1. Two equally available adapter learning rates .003/.01. DEV checkpoint selection at0 and every100 updates.
- Supervised arms: unadapted longer loop (no fitting); depth-shared LoRA rank2; per-loop LoRA ranks1/2; per-loop FiLM; per-loop two-Householder Mirror; Mirror+LoRA rank1. All inserted functions initialize to identity/no update. Mirror is Q-transpose GELU(Qz), Q=H(v2)H(v1), v2=v1 initially, inverse in reverse order. Attention is not adapted.
- Distillation: selected supervised per-loop rank2 LoRA teacher frozen; collect TRAIN/DEV FFN inputs and outputs. Refit LoRA rank1/rank2, FiLM, Mirror, Mirror+rank1 using identical frozen source. Minimize per-loop FFN error normalized by TRAIN teacher correction energy. Select solely by DEV state error, not task labels or test. Local teacher-forced fit and autonomous rollout quality reported separately.
- Expected main fitting:5 source,120 supervised adapter,100 local distillation fits;10 no-adaptation extended baselines.235 candidate checkpoints,125 DEV-selected entries. Preserve all candidates.

## Predeclared judgments
Primary depth8; depth4 secondary. LoRA repair gate: at least10% lower TEST CE than unadapted8 and no accuracy loss in at least4/5 paired seeds. Compare to source2 separately: repairing extension is not beating the source model.

Mirror replacement gate (all required, >=4/5 seeds): distilled Mirror retains rank2 teacher accuracy within1 percentage point; CE no more than1.05x teacher; recovers at least80% of teacher-forced local correction relative to no adaptation; actual adapter serialization no more than50% of rank2 teacher. Near-zero correction denominators are not evidence of success.

Mirror specificity gate (>=4/5): local reconstruction error <=.90 of same-code FiLM; rollout CE <=1.05x FiLM. Actual full model bytes and batch1/64 latency reported separately; equal optimizer updates are not equal FLOPs. Rank2 refit is a reconstructability control.

Predetermined additional diagnostics: reverse adapter order; compare per-loop vs depth-shared LoRA with unequal capacity explicitly reported; TEST Gaussian pixel noise std.15, seed1234567, clamp[0,1] (corruption sensitivity, not external OOD).

## Pilot disclosure
One DEV-only seed9001, excluded from the main five. Source2 DEV CE .08383, accuracy98.05%; unadapted8 CE1.6568, accuracy66.30%; adapted8 LoRA2 CE.12760, Mirror.14366, FiLM.12724. This identifies a nontrivial loop-extension deficit. No TEST metric observed. First pilot tool call stopped mid-run; completed source/LoRA retained, remaining arms resumed identically. All logs retained.

## Frozen provenance before main run
Local isolated git commit: aa4277d147b058b95851862d6680227e90617fd0.
PROTOCOL.json SHA256: 700b6333ecf98d9a37f967ca9aa4cfc093c4e6358493811b0c6bd2dc4656f0d9
SOURCE_FREEZE.json SHA256: 58f28fc1d134106fa1c63a979952627d94f240fad6acad969e34144e651df0ff
source/core.py SHA256: 41666e84a3af14766d3892c8d499333b9c240c7c3e915846f378ba23bfca86c6
source/engine.py SHA256: 14ac1003dfe82b15ea4e82518dcb1b9739f8c60016ee92e3d5aca68b9f904caf
source/run.py SHA256: 0f6946c598c3743f424aa4ef51b3ef0036cc7f6f676527f682a6de162039a84f
14 source/mechanism tests passed. CPU FP32 eager; no GPU. All original code, data splits, checkpoints and logs will be included in the conversation artifact after verification. No main merge or universal Mirror advantage is authorized by this record.
