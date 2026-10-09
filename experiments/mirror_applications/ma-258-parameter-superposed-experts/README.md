# MA-258 — Parameter-superposed expert bank with Mirror unbinding

**H:** One shared expert matrix plus per-expert Givens coordinates can replace copies for aligned expert views; independent expert residuals may require private state.

**T:** Eight 32×32 expert functions in aligned Givens and independent rank-2 strata. Compare untied experts, hard tying, native random-sign PSP, one-angle Mirror and rank-2 residual controls. Expert ID supplied to isolate storage. Fresh worlds 25810–25812 × seeds 0–2; actual serialized bytes charged.

**D:** Pending.

**C:** Rank-2 private residuals can represent the Givens orbit and may approach Mirror storage; PSP unbinding may incur cross-talk.

**U:** Fresh quality, bytes and inference cost in both strata.

## Development audit before fresh

An initial dev pass had a transpose mismatch and unconstrained periodic angle optimizer. The untied exact-reference test caught the mismatch before fresh. The runner was corrected to use x@W consistently and to constrain angle fitting to the known generator interval [-0.7, 0.7]. Original exploratory summary (not authoritative): Mirror/tied/untied errors clustered around 1.37 in aligned tasks; after fixing orientation, bounded Mirror fit achieved near-exact aligned reconstruction. The original faulty CSV was overwritten by the corrected development-only run; this protocol amendment preserves the known exploratory outcome and cause.
