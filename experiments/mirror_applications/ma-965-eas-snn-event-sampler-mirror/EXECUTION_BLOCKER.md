# MA-965 execution scope record

Status: **NOT ESTABLISHED / deferred**, registry remains **UNTESTED**. This is not a scientific FAIL.

## Checked facts

- The official source exists: `Windere/EAS-SNN`, upstream HEAD `423cb8be283af75426c908e94b723456f961913d` (Apache-2.0).
- `yolox/models/embedding.py::AdaptiveRSNNEmbedding` is ordinary PyTorch and implements recurrent convolutional adaptive sampling, spike-triggered segment assignment, and `sum`/`last`/`avg` readout.
- The official README lists Gen1, 1Mpx, and N-Caltech datasets and pretrained checkpoints. No matching data/checkpoint paths were found under `/workspace` (bounded search to depth 4).
- `torch.cuda.is_available()` is false and `nvidia-smi` is unavailable. The published detector training command uses four GPUs; the upstream event evaluator contains CUDA-specific paths.

## Why this candidate is deferred

The registered primary outcomes include detection mAP, event latency, sample count, and energy on a real event dataset. No local event dataset or trained checkpoint is available, and the current environment has no GPU. A newly invented synthetic sampler task would not validate that registered detector claim, so this worker skips the full MA-965 experiment instead of substituting a proxy or changing its scope after seeing results.

## What would unblock it

Make a pinned official Gen1 split and the official EAS-SNN Gen1-S checkpoint available, plus a GPU environment for the native training/evaluation path (or an audited CPU evaluator if runtime is excluded). Then freeze the native shared sampler, direct threshold/gain, equal-byte FiLM/low-rank, Mirror coordinate, and independent-sampler controls; reserve whole sequences and sensor/time conditions for fresh evaluation. Charge complete detector payload bytes and report event timing/sample coverage. Spike counts alone are not energy.

## Provenance

- Paper: https://arxiv.org/abs/2403.12574
- Official implementation: https://github.com/Windere/EAS-SNN
- Pinned upstream commit: `423cb8be283af75426c908e94b723456f961913d`
- Local checkout: `59fa0e0d3` (MA-964 verification head)
