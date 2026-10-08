# MA-418 — compositional latent factors

Status: SCREENING; protocol frozen before runs. Prior art PA68 (DeepSDF) and PA79 (concept modulation).

## H — Hypothesis
Object, style and domain latent factors trained on incomplete combinations can compose to reconstruct unseen combinations with lower actual payload than independent per-combination latents.

> **Mirror insertion:** this experiment adds factor codes `m(o,s,d)=Eo[o]+Es[s]+Ed[d]` before a shared neural decoder, so unseen combinations can reuse factor views without storing a full latent vector for every combination.

## T — Setup
A single frozen `Fθ(x,z)` MLP is shared identically. The 8×8×4 combination grid is split 80/20 with every factor value represented in training. Compare additive factor tables, independent DeepSDF codes with heldout support adaptation, a native additive factorization with the same algebra, and oracle codes. Query data for heldout combinations are disjoint from 128 support points. Actual decoder-plus-code NPZ bytes are authoritative.

Fresh 41811–41813 remain sealed unless both development seeds pass the frozen quality and byte gates.
