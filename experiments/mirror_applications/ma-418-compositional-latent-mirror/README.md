# MA-418 — compositional latent factors

Status: **FAIL at the frozen development screen; fresh seeds remain sealed.** Prior art PA68 (DeepSDF) and PA79 (concept modulation).

## H — Hypothesis
Object, style and domain latent factors trained on incomplete combinations can compose to reconstruct unseen combinations with lower actual payload than independent per-combination latents.

> **Mirror insertion:** this experiment adds factor codes `m(o,s,d)=Eo[o]+Es[s]+Ed[d]` before a shared neural decoder, so unseen combinations can reuse factor views without storing a full latent vector for every combination.

## T — Setup
A single frozen `Fθ(x,z)` MLP is shared identically. The 8×8×4 combination grid is split 80/20 with every factor value represented in training. Compare additive factor tables, independent DeepSDF codes with heldout support adaptation, a native additive factorization with the same algebra, and oracle codes. Query data for heldout combinations are disjoint from 128 support points. Actual decoder-plus-code NPZ bytes are authoritative.

Fresh 41811–41813 remain sealed unless both development seeds pass the frozen quality and byte gates.

## D — Outcome
Both development seeds missed the held-out quality gate. Factor-code NRMSE was **0.3323 / 0.2918**, against support-adapted independent DeepSDF at **0.1676 / 0.1510**. Actual payload was **13,286 / 13,312 bytes** for factor codes and **19,244 / 19,209 bytes** for DeepSDF (0.691 / 0.693x); storage passed, quality failed. Factor throughput was 4.11M / 4.52M logical queries/s versus 5.01M / 4.89M for DeepSDF. The exact native additive control matched the factor predictions; factor payloads are one byte larger due to method metadata. Oracle NRMSE was 0.00577 / 0.00168.

## C — Strongest counter-hypothesis
The teacher itself is additive in latent space and the oracle table is accurate, so this run does not show a representation-capacity barrier. The fixed 600-update factor fit may be under-optimized or poorly conditioned. Because native additive factorization is algebraically identical, Mirror-specific value is not established even if a future optimizer closes the gap.

## U — Still unconfirmed
Near-convergence capacity, natural task transfer, alternate factor geometries, and stronger optimizer schedules. Fresh seeds were not accessed after the failed gate.

**Evidence labels:** Facts are the measured numbers above; the FAIL is an interpretation under the frozen fixed-budget gate; optimization/identifiability is a hypothesis.
