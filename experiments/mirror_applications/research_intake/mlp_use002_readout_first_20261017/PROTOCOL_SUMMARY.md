# MLP-USE-002 frozen design

The byte-identical executed `PROTOCOL.json`, source and SHA freeze are in the conversation's source ZIP; this text is a reviewable research summary, not a substituted preregistration.

**H**: factorized gate/up Givens coordinates improve on existing task-specific output readout and equal 32-param native gain/coded-pair alternatives.

**T**: real-digit sklearn dataset of 1,797 observations. Two roles, four sub-tasks each. New splits 90831–90833, seeds 11/22, models `pair_readout`, `pair_readout_mirror`, `pair_readout_gain`, `pair_readout_pair_mirror`. Strong parent task-readout trained in all conditions. Training 700 steps/AdamW(.012,WD.01)/batch64, checkpoints140/350/700. Validation checkpoint chosen without test. Development world90800 used only for smoke/calibration.

**D**: primary joint gate <=0.95x paired average loss and <=1.05x worst-task loss in >=2/3 new worlds. For native readout also <=1.10x physical serialized bytes. All three required quality gates fail in both roles (attribute 1/3, denoise 0/3). Four exact deterministic retrainings, 96 FP64-readback-verified inference files.

**C**: extra Givens code cannot help when a linear task-specific down head already captures the needed difference. Source-loss differences from training initialization are controlled by paired seeds.

**U**: 3 splits of 1 real corpus, not independent datasets; digit attributes share one label; unoptimized CPU eager kernels; not a converged/exact capacity bound; no NLP/Transformer/GPU conclusions.

ERROR CHECK: ratios of group-wise medians differ from medians of *paired* world ratios. Only paired world values support pass/fail decisions.