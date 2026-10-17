# MA-1164 — Mirror K/R scaling against fair pretrained NN controls

**Mirror insertion:** substitute an m-dimensional physical coefficient vector and low-description K-direction/R-round nonlinear Mirror program for the 1,210 independently fitted FP64 weights of a trained 64→16→10 MLP, then evaluate task preservation and cost, without using heldout labels for compression.

Hypothesis: changing K or R while holding m and address selection fixed can improve the heldout task-quality / inference-byte frontier beyond the same-m **non-Mirror orthogonal sparse code** and a **byte-near low-rank SVD** baseline.

Prior-art/direct controls: MA-1163 nonlinear Mirror compiler, orthogonal sparse inverse coordinate basis, native FP16/FP32 storage, SVD first-layer compression, direct target-trained MLP quality ceiling. The single-scalar case is *not* the experimental focus: m=64/256/512/768 (development), m=512/768 (fresh).

This is a follow-up study with **new** split seeds and distinct held-out partitions. Previous MA-1163 fresh worlds 618310..312 were opened and are not reused. The Mirror transforms and nonlinearity have no learned auxiliary matrix; address indices are physically stored and counted. Original baseline pretrained weights are trained first, fixed before conversion, and compressed using **weight error only**. Optional adaptation from training images is a separately identified analysis.

Success gate: fresh paired Mirror model beats the strongest *byte-near* native non-Mirror control by ≥1 percentage point in ≥2/3 worlds, retains baseline accuracy within 2 percentage points, and reduces full inference payload including model-specific metadata. Failures and controls are retained.

Only a substantial improvement above non-Mirror controls would justify widening the Mirror family or specialized optimizer; K and R by themselves do not create independent stored degrees of freedom. No claim of float16 GPU wall-time or parameter-count multiplication.
