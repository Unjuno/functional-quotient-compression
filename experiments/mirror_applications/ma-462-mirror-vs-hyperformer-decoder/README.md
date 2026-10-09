# MA-462 — Mirror decoder vs HyperFormer decoder under fixed embeddings

Status: **FAIL for Mirror-specific advantage; scoped aligned quality/compute result**
Protocol A1 freeze: `7af3b858`; original fresh worlds 46210–46212; A1 control worlds 46220–46222.

## H — Hypothesis

With the same fixed task/layer/position embeddings, a structured Mirror decoder produces adapters at least 30% smaller than a generic HyperFormer decoder without more than 10% query NRMSE loss.

## T — Test

Used 2×2 adapters with fixed four-value context embeddings. The teacher contains an aligned rotation orbit for three quarters of contexts plus off-orbit residual directions for one quarter. Compared the Mirror decoder `R(c0) W R(c1)`, a 4→16→4 HyperFormer MLP, an affine decoder, a rank-two basis decoder, and independent matrices. Development selected 1,500 MLP updates. Three fresh worlds × three seeds; 64 training and 32 held-out contexts per run. Actual packages include embeddings and all decoder state.

The initial fresh result showed Mirror ahead of the MLP. Amendment A1 added a fixed Fourier-feature linear decoder as a targeted generic control and used new worlds; initial results remain separately recorded. The Fourier decoder uses features `[cos(c0)cos(c1), cos(c0)sin(c1), sin(c0)cos(c1), sin(c0)sin(c1), c2, c3, 1]` and the same embeddings.

## D — FAIL for Mirror-specific claim

**Fact:** On original fresh worlds at N=32, Mirror averaged NRMSE 0.0691 and 73.16B/context versus the HyperFormer MLP's 0.1991 and 110.78B/context. This was a quality/bytes win over the selected fixed-budget MLP.

On A1 fresh worlds, Mirror averaged NRMSE 0.0663 and 73.16B/context. The Fourier decoder reached 2.55e-7 at 77.16B/context. Mirror uses 94.8% of Fourier payload bytes and has much worse quality, failing the A1 Mirror-specific gate. The affine and rank-two controls had NRMSE near 1.0; the independent upper bound was exact. Mirror decoder compute was 16 MAC/context vs 28 Fourier features and 128 MLP MACs, so the aligned Mirror result retains a compute advantage over those generic decoders.

**Interpretation:** Mirror's direct angle decoder is an effective inductive bias for the aligned rotation subgroup and beats the trained MLP at lower bytes and compute. A fixed generic Fourier basis reproduces the same teacher almost exactly with only 5.2% more bytes, so the evidence does not establish a Mirror-specific storage frontier. The off-orbit quarter also prevents exact Mirror fit.

## C — Strongest counter-hypothesis

The task family explicitly contains the Mirror rotation orbit; Fourier features are the natural generic decoder for that same geometry. The result may be explained by injecting the correct trigonometric features rather than by Mirror-specific capacity.

## U — Unknown

Natural Transformer adapters, learned embeddings, larger dimensions, and task families whose geometry is not known in advance remain untested. This is a synthetic decoder mechanism screen only.
