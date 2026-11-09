# MA-751 — ControlNet branch compressed into Mirror views

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

On eight synthetic spatial conditions, copied branches achieve nRMSE `4.0e-8` at 1,617 B. A shared hard-tied branch uses 1,085 B but nRMSE `.219`; output FiLM uses 1,866 B with `.166` error. Givens View preserves output to numerical zero but uses 1,692 B and 4,194,304 active MACs, versus 1,617 B and 3,407,872 MACs for copied branches. The native direct-Givens control exactly matches Mirror payload hash, bytes, error and MACs.

## Interpretation

The role View recovers condition-specific spatial control in this intentionally aligned branch family, while hard tying loses fidelity and FiLM is insufficient. The actual serialized copied branches are smaller than the shared Givens state, and View reconstruction adds about 23% MACs. The method is an ordinary Givens-conditioned branch, so no Mirror-specific value is established.

## Hypothesis result

**FAIL** for the joint byte/compute gate and Mirror-specific value. Fresh/audit was not opened after the exact native alias and byte miss.

## Frozen protocol / scope

PA190. A two-layer convolutional spatial control branch (1→8 channels, 3x3 conv, ReLU, 8→4 pointwise conv) is evaluated on eight condition IDs. The teacher applies six Givens rotations to the first four hidden channels before the output convolution. Compare copied output branches, hard tying, train-fit per-channel FiLM, native direct Givens and Mirror. Train maps fit FiLM, heldout maps evaluate spatial nRMSE. Full serialized NPZ bytes and a convolution/view MAC proxy are charged.

This does not include a pretrained diffusion model, ControlNet training, prompt conditioning, image generation quality or hardware latency.

## H / T / D / C / U

- **H:** Shared control branch plus low-description condition Views reduces branch storage while preserving spatial control fidelity and compute.
- **T:** Seed 75101; eight conditions; 32 train/16 heldout maps per condition; 16x16 fields; independent, hard-tied, FiLM, native Givens and Mirror controls.
- **D:** **FAIL.** Mirror is 1,692 B vs 1,617 B copied; MACs are 23% higher; native Givens is an exact alias.
- **C:** Givens View recovers aligned condition transforms but requires extra compute, and compressed copied branches are already smaller.
- **U:** Pretrained ControlNet, generated image quality, real conditions, learned View training and accelerator inference.
