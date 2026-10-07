# Mirror Parameter Integration Matrix

Date: 2026-10-08 JST  
Status: active breadth-first validation map

Read [MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md](MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md) first.

The common operation is:

`F(x; theta) -> F(x; theta, m)`

The table below tells workers where to try the same extra low-description functional parameter `m` in different established methods.

| Family | Native physical object | Minimal Mirror insertion | Mirror question | Strong direct control |
|---|---|---|---|---|
| MoE | expert FFNs / expert pool | expert/layer View `m` around shared/reused expert | fewer physical experts for useful logical specialization? | tied expert + embedding, BatchEnsemble, MoRE/UniPool |
| LoRA / adapters | adapter matrices / low-rank basis | task/layer coefficient or transform `m` | many logical adapters from fewer physical bases? | LoRA, VeRA, IA3, AdapterFusion |
| Attention heads | Q/K/V/O projections | head/role `m` on shared projection | fewer physical head projections? | MQA/GQA, QKV sharing |
| KV cache | canonical K/V or latent cache | view-specific key/value/readout `m` | switch logical views while reusing physical cache? | MLKV, MLA, ordinary cache duplication |
| Tied depth | recurrent/shared block | layer/step `m` | recover layer-specific function without new block weights? | ALBERT/Universal Transformer, timestep embedding, static LoRA |
| FFN roles | shared nonlinear block | hidden-feature / function `m` | multiple logical nonlinear roles from one MLP? | gate, IA3, CondConv |
| KAN | parent/basis edge function | edge/task transform `m` | many edge functions from shared physical function bank? | standard KAN, GS-KAN |
| Embeddings | shared embedding table/basis | domain/token/source `m` | domain-specific logical embeddings without tables per domain? | factorized embeddings, hash embeddings |
| SSM / recurrent dynamics | state transition/read/write operator | task/role/dynamics `m` | logical dynamical experts from shared transition basis? | S4/Mamba native conditioning |
| Fast weights / DeltaNet | writable matrix state | low-rank/state/update-rule `m` | smaller writable state for same online function? | full Hebbian/DeltaNet state |
| Plasticity | slow weights + plasticity rule | task/context plasticity `m` | multiple online learners from one physical network? | differentiable plasticity, learned optimizer |
| Representation steering | activation/intervention basis | behavior/condition `m` | many behaviors from one intervention basis/chart? | ReFT, linear steering, INNSteer |
| Function vectors | causal function-vector basis | task/composition `m` | compact reusable function library? | explicit function vectors |
| Model editing | edit basis / memory bank | fact/edit `m` | more edits per stored physical basis? | ROME, MEMIT, GRACE, SERAC |
| AI Engram | causal memory traces | memory coefficient `m` | causal memory compression and reversible manipulation? | raw engrams, edit-vector baselines |
| Task vectors / merging | task delta basis | task/composition `m` | logical task models from compact shared delta atoms? | Task Arithmetic, TIES, DARE, KnOTS |
| Model manifold | learned subspace/surface | manifold-position `m` | useful logical models addressed by few coordinates? | subspace training, Model Soups |
| Ensembles / posterior | member/posterior subspace | member/sample `m` | diversity/uncertainty without model copies? | BatchEnsemble, SWAG, Packed Ensemble |
| Federated personalization | global model / shared adapter basis | client `m` | many clients with small persistent state? | pFedHN, client LoRA |
| Optimizer geometry | gradient/update subspace | task/layer update `m` | reusable optimization rules/subspaces? | GaLore, Meta-SGD |
| Neural operators | operator basis | PDE/boundary/regime `m` | many physical-system operators from one basis? | FNO, DeepONet, KNO |
| Relational GNN | relation transform basis | relation/schema `m` | many logical relation transforms? | R-GCN basis decomposition, CompGCN |
| Diffusion control | control branch/adapter basis | control/depth/timestep `m` | replace banks of control adapters/branches? | ControlNet, T2I-Adapter, Ctrl-Adapter |
| Robot policy | policy/controller/option adapter | skill/embodiment `m` | add skills without full policy/adapter copy? | option adapter bank, LoRA policy |
| Neural cellular automata | local update rule | goal/rule/damage `m` | many global behaviors from one local rule? | GoalNCA, Attention NCA |
| Programmable graph | reusable node functions / substrate | topology/program `m` | logical architectures as compact state? | GrapNet, routing networks |
| Collective inference | private agents + communication policy | admission/coalition `m` | task-specific collective protocols without model sharing? | Mesh Inference native policy |
| Packet / multi-token decoding | future head / packet state | slot/branch/phase `m` | multiple future roles from fewer heads/state? | PTP / explicit slot heads |
| Quantization / gauge | quantized shared weight | gauge/codebook `m` | lower rate/error through addressable gauge? | QuaRot / codebook quantization |
| Supernets / elastic models | shared supernetwork | width/depth/architecture `m` | cheap correction for many submodels? | OFA, MatFormer, US-Net |

## Required variants after a direct screen

A family that passes a first mechanism screen should usually be expanded in this order:

1. **static `m`** — one learned/stored code per logical function;
2. **factorized `m`** — e.g. task x layer, expert x depth, condition x behavior;
3. **dynamic `m(x,h)`** — generated from context/history when justified;
4. **shared + private residual** — quantify the part of natural variation that does not fit `m`;
5. **composition** — signed/sparse/product/sequential combinations only after individual functions work.

Do not jump directly to the most complicated variant.

## Breadth priority

Prefer families with:
1. a clearly duplicated physical object;
2. a cheap native baseline;
3. a low-cost insertion point for `m`;
4. a measurable storage/state/cache benefit;
5. a controlled aligned test plus a natural/misaligned test.

The purpose is to map where the **same Mirror parameter principle** survives contact with different architectures.
