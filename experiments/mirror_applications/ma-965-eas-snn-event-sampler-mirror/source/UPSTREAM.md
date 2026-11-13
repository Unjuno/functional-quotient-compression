Official implementation source inspected (not vendored):

- Repository: https://github.com/Windere/EAS-SNN
- Commit: `423cb8be283af75426c908e94b723456f961913d`
- Relevant module: `yolox/models/embedding.py`, `AdaptiveRSNNEmbedding` (lines 79–226 at this commit)
- License: Apache-2.0
- Core structure: recurrent gate/current convolutions; sigmoid gate; threshold/reset; spike-triggered assignment into `Ts` segments; configurable sum/last/average readout.
