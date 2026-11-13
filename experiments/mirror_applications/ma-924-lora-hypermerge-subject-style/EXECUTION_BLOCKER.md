# MA-924 execution blocker

## Fact

- Official repository `SamsungLabs/LoRA.rar`, inspected commit `54ae5578702ad9856cfb919ae2a4a2f487cf0cc6`, contains `models/hypernet.pth` and `models/koala_700m_hypernet.pth` (~1.9 MB each).
- It does not include subject/style LoRA weights or test images. README directs users to download source images and train all subject/style LoRAs before evaluating held-out combinations.
- `inference_lorarar.py` loads Stable Diffusion XL and hardcodes `pipeline.to(device="cuda", dtype=torch.float16)`.
- Runtime PyTorch is CPU-only (`torch.cuda.is_available() == False`).

## Decision

Keep UNTESTED. Random/synthetic LoRA tensors could test hypernetwork tensor plumbing or merge latency only, not subject/style quality or the native LoRA.rar comparison. Do not relabel such a smoke test as the registered result.

## Resume condition

Provide GPU-capable SDXL inference plus subject/style LoRA assets and held-out references. Freeze subject/style combinations before fresh evaluation; count all hypernetwork, LoRA, Mirror code, bases and merged deployment weights in actual bytes.
