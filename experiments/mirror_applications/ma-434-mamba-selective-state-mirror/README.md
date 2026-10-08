# MA-434 — role-conditioned Mirror views for a selective SSM

## H — falsifiable hypothesis

One shared input-selective state-space block plus one small role Givens code will recover role-specific sequence dynamics at lower actual bytes than independent role SSMs while keeping the same recurrent compute.

## T — protocol

Synthetic four-role sequences with input-dependent time steps and role-specific rotated state dynamics. Compare shared selective SSM, Mirror role angles, role-specific scalar Δ gates, and independent role SSMs. Train on sequences and evaluate held-out sequences. Measure sequence prediction error, actual serialized state bytes, active recurrent compute, and stability.

This is a small Mamba-like mechanism proxy, not a Mamba benchmark reproduction.

