# MA-258 — Parameter-superposed expert bank with Mirror unbinding

**H:** One shared expert matrix plus per-expert Givens coordinates can replace copies for aligned expert views; independent expert residuals may require private state.

**T:** Eight 32×32 expert functions in aligned Givens and independent rank-2 strata. Compare untied experts, hard tying, native random-sign PSP, one-angle Mirror and rank-2 residual controls. Expert ID supplied to isolate storage. Fresh worlds 25810–25812 × seeds 0–2; actual serialized bytes charged.

**D:** Pending.

**C:** Rank-2 private residuals can represent the Givens orbit and may approach Mirror storage; PSP unbinding may incur cross-talk.

**U:** Fresh quality, bytes and inference cost in both strata.

## Development audit before fresh

An initial dev pass had a transpose mismatch and unconstrained periodic angle optimizer. The untied exact-reference test caught the mismatch before fresh. The runner was corrected to use x@W consistently and to constrain angle fitting to the known generator interval [-0.7, 0.7]. Original exploratory summary (not authoritative): Mirror/tied/untied errors clustered around 1.37 in aligned tasks; after fixing orientation, bounded Mirror fit achieved near-exact aligned reconstruction. The original faulty CSV was overwritten by the corrected development-only run; this protocol amendment preserves the known exploratory outcome and cause.

## Results

**D: PROMISING, tightly scoped to the aligned synthetic orbit.** In all three fresh worlds × three seeds, the one-angle Mirror view reconstructed the aligned expert family at mean NRMSE 0.00041 (mean worst-expert NRMSE 0.00302) using 5,925 serialized bytes. Untied experts used 34,409 B; packed rank-2 residual factors used 10,209 B at exact output reconstruction. Hard tying used 5,737 B but had NRMSE 0.07776. Mirror therefore kept aligned quality at 82.8% fewer bytes than untied and 42.0% fewer than the exact rank-2 control.

For independent rank-2 expert residuals, Mirror NRMSE was 0.23828, close to hard tying at 0.22333; packed private rank-2 factors recovered exact outputs at 10,209 B, still below untied storage. Native PSP with deterministic context regeneration used only 5,801 B but had NRMSE 2.62 in both strata, showing severe unbinding interference.

**Fact:** 90 fresh rows, 9 worlds/seeds per stratum-method grid, pinned generator and payload hashes. A2 serializer/decode replay left all quality values and predictions identical to the first fresh run. Mirror coordinate fitting averaged 1.56 s/bank in aligned and 1.48 s/bank in random tasks; Givens weight reconstruction ~0.23 ms, eight-expert application ~0.12 ms. Rank-2 factor decode averaged ~0.07 ms.

**Interpretation:** A low-description view can replace physical expert copies when functions lie on the exact Givens orbit it encodes. Independent rank-2 variation leaves that orbit; private residual factors restore quality. In this synthetic task, even the exact private rank-2 control is smaller than untied. Native PSP unbinding alone was not useful.

**C:** The aligned stratum is generated from the exact Mirror family, a feasibility case. A generic structured basis, learned adapter, or different orthogonal parameterization may match it; no natural MoE routing or language result was tested.

**U:** Natural experts, learned routers, larger matrices, non-aligned views, and runtime-fused kernels remain untested. This is not evidence for broad MoE adoption.
