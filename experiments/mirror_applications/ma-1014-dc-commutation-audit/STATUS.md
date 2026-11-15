# MA-1014 status

Stage: COMPLETE — FAIL (bounded synthetic mechanism screen).
Branch: `research/ma-1014-dc-commutation-audit-20261115`.
Base: worker-ready baseline `research/mirror-application-worker-ready-20261007`.
Fresh seeds: locked and unopened.
Last verified commit: `9d000dc09e492a381faaefe2a8667b95a207c337`.
H: placement of the feature Mirror view relative to strict MRI DC changes whether acquired measurements are preserved, and projected views may improve heldout quality.
T: one frozen 32×32 two-coil prior; 64 updates on 16 source phantoms; three target sites × eight phantoms; five placement controls; fresh seeds sealed.
D: FAIL. Post-DC view violates acquired consistency; projection restores it but harms heldout quality. Pre-prior view remains within <0.5% of native prior and misses 5% gate at all sites, while adding 64 B.
C: fixed angles/short source training may be poorly aligned.
U: acquired multicoil MRI, spatially varying SENSE projection, pretrained MoDL/VarNet/DUNE and fresh-seed validity.
