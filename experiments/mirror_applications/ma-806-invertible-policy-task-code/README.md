# MA-806 — One-step invertible policy with skill coordinates

Status: **FAIL for Mirror-specific value** (synthetic policy screen)

## Prior-art boundary

PA202 puts an invertible adapter around one-step flow-matching actions. This screen uses an orthogonal shared policy and two skill-specific Givens rotations. It tests one-step action accuracy, storage and the compute cost of four-step straight-line flow integration. No robot simulator, manipulation data, trained INN adapter, or robot hardware is involved.

## Fact

Seed 80601; 8D observations/actions; nine skills; 256 training and 512 test examples/skill; zero optimizer updates. Independent fitted invertible policies use 2,235 B with action nRMSE 9.33e-8, success fraction 1.0 and orthogonality error 3.46e-7. Shared policy without skill code uses 868 B but nRMSE .1008 and success .066. Shared base plus two Givens angles uses 1,346 B (39.8% less than independent), nRMSE 1.02e-7, success 1.0 and orthogonality error 4.12e-7. The native Givens control has the identical payload hash and behavior. Givens application costs 76 MAC/action vs 64 for the independent dense matrix. Four straight-line flow evaluations retain nRMSE near 1e-7 but require four policy evaluations instead of one; CPU timings are small synthetic calibrations only.

## Interpretation

Skill codes recover the nine synthetic task maps while keeping the adapter invertible and one-step. The code compresses storage against independent policies, but it is exactly ordinary Givens angle parameterization, and the shared policy-only control shows that skill-specific variation is needed. Iteration adds compute without improving this straight-line target. No Mirror-specific or robotic capability gain is established.

## H / T / D / C / U

- **H:** Shared invertible action geometry plus compact skill coordinates preserves one-step success and reduces full per-skill storage without iterative denoising.
- **T:** Seed 80601; orthogonal 8D base, two Givens planes, nine skills; 256 train/512 test examples per skill. Compared independent fitted policy matrices, shared-only, Mirror Givens codes, native Givens control, and four-evaluation flow sampler. Measured actual NPZ bytes/hash, action nRMSE, success at fixed tolerance, orthogonality, MACs, flow calls, timing and replay.
- **D:** **FAIL** for Mirror-specific value: 39.8% fewer bytes than independent and one-step success 1.0, but the native Givens control exactly aliases the View; Givens adds 18.75% active MACs. Four-step flow integration has the same quality and four evaluations.
- **C:** Standard per-skill Givens adapters over a shared orthogonal policy fully explain the compression and function.
- **U:** Real manipulation, multimodal action distributions, trained flow matching, adaptation from demonstrations, collision/safety metrics, robot latency and hardware.

Fresh remained sealed after exact native alias. No robot-task claim is made.
