# MA-504 — Token-conditioned Mirror ReFT

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE  
Base commit: `e87c50a3` (verified through MA-503)  
Prior art: PA96 (ReFT/LoReFT), PA63 (FiLM)

## Hypothesis

H: For continuous token contexts, a rank-4 LoReFT intervention with a small context-generated Givens View can recover the tokenwise intervention with fewer serialized bytes than FiLM and a generic coordinate MLP, while preserving intervention quality.

## Mirror insertion

> **Mirror insertion:** this experiment adds token-conditioned Givens coordinates `m(u_t)` to a shared rank-4 LoReFT activation basis, so each token receives a distinct intervention without storing an independent intervention vector per token/context.

- Shared object: fixed synthetic 64×4 orthonormal intervention basis and one shared rank-4 projection from token activations.
- Context: continuous 8D vector supplied for each token.
- Mirror coordinate: two Givens angles predicted from the token context; rotate the projected activation code before decoding through the shared basis.
- Native LoReFT control: direct linear low-rank intervention generator from `[h_t,u_t]`.
- Required simple controls: shared/no intervention, context-conditioned FiLM affine modulation, generic MLP producing rank-4 LoReFT coefficients.
- This directly addresses MA-403's counter-hypothesis that four discrete contexts make a static angle table cheaper: contexts here are continuous, independently sampled, and unseen at test.

## Frozen protocol

- Token hidden size 64; context size 8; intervention rank 4.
- Each target token intervention is `delta(h,u)=B R(Wu) P h`; output is `h+delta`. B is fixed and shared; its bytes are charged to all ReFT methods. The generator/task target is synthetic and exact by construction.
- Development worlds 50400/50401, seeds 0/1/2; select only learning rate from `{0.003,0.01}` using mean validation intervention NRMSE across methods.
- Fresh worlds 50410/50411/50412, seeds 0/1/2; locked before fresh; one selected learning rate only.
- 8,192 train tokens, 2,048 validation tokens and 4,096 test tokens per world; 1,000 AdamW updates, batch 128.
- All methods receive the same hidden/context inputs and target examples.

## Controls

1. Shared no intervention.
2. Linear LoReFT: direct rank-4 coefficients from `[h,u]`.
3. Mirror Givens: shared projection `P h`, two context-generated angles `W u`, Givens rotation, shared basis decode.
4. Generic MLP LoReFT: MLP over `[h,u]` outputs rank-4 coefficients.
5. FiLM: context-conditioned per-channel scale and bias on the token activation.

## Gates and accounting

- PASS if Mirror test intervention NRMSE ≤0.05 in every fresh world, uses ≤80% of FiLM payload bytes, and matches or beats the generic MLP LoReFT within 10% relative NRMSE.
- PROMISING if quality/storage pass but runtime or a generic simpler control dominates a Pareto axis.
- FAIL if a quality/storage gate fails or a simpler control matches quality at equal/fewer actual bytes.
- Serialize all inference parameters, basis, context generator, and metadata with `torch.save`; report exact payload length and SHA-256.
- Report training examples/tokens, optimizer updates, MAC proxy/token, measured CPU throughput, training wall time, test intervention NRMSE and output NRMSE.

## Boundaries

Synthetic continuous-context intervention only. This is not a pretrained Transformer or language task, not capacity evidence, and not a general claim about all token-conditioned ReFT. The teacher is aligned to a Givens View; stronger simple controls remain decisive.
