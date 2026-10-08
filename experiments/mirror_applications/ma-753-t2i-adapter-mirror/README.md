# MA-753 — Mirror compression for composable image-latent controls

Status: FAIL
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA191 — T2I-Adapter

## H — falsifiable hypothesis

A shared adapter matrix with a small control-specific Givens coordinate can represent multiple aligned control adapters and held-out additive compositions with less actual serialized state than independent adapters. Compare with unrestricted two-basis synthesis before making a Mirror-specific claim.

## T — frozen scope

This first screen isolates the adapter mechanism in a synthetic 8D latent regression because the available container is CPU-only and no pretrained diffusion checkpoint is provisioned. Four control types are trained individually; all six two-control sums are held out for audit. The aligned world uses four Givens views of one matrix; the boundary world uses four independent matrices. Controls are independent adapters, unrestricted two-matrix basis mixing, per-control FiLM, and hard sharing. Exact conditions/seeds/gates are in `PROTOCOL.json`; Draw30 replay is in `source/draw30_exclusions.json`.

## D — decision

**FAIL for the preregistered aligned quality gate; the storage subgate passes.** Across 3 fresh worlds × 3 model seeds, aligned Mirror held-out pair MSE averaged `7.16e-5`, while independent adapters were at numerical zero (`2.65e-14`); the relative 1.10× quality gate therefore failed in all 3/3 worlds. Mirror used 440 serialized bytes versus 1,192 B for the independent adapter bank (36.9%, passing the <=60% storage gate). The ordinary two-basis mixture was also effectively exact (`2.97e-14`) at 712 B, only 272 B more than Mirror. Thus Mirror offers a smaller approximate point on this constructed teacher's storage/quality frontier, but does not match independent quality or establish a unique function family.

### Fact

On the independent-matrix boundary, Mirror pair MSE averaged 1.305; independent adapters remained at `2.04e-14`, and basis_mix2 averaged 0.765. Mirror payload/latency were 440 B/0.263 ms; basis_mix2 712 B/0.019 ms; independent 1,192 B/0.0066 ms. The Mirror implementation eagerly materializes all four rotated matrices before selecting a control, which contributes overhead. This CPU latency is an implementation measurement, not an optimized kernel comparison.

### Interpretation

Aligned coordinate structure can reduce adapter bytes substantially, but the native independent bank and ordinary low-rank basis preserve the teacher more accurately. Unrelated functions need private or richer state. Mirror is much slower in this eager implementation, and the two-basis control reproduces the aligned teacher almost exactly with a modest payload increase.

### Hypothesis

The aligned result is expected because the teacher was generated from the same Givens family. Whether the 272 B reduction over ordinary rank-2 basis mixing is useful in a trained diffusion model remains unknown. This fixed-update screen does not establish increased capacity.

All 540 pair-level audit rows and serialized payload hashes are retained. The audit was replayed: task IDs, payload bytes, and query errors matched exactly; measured timing varied.

## C — strongest counter-hypothesis

The teacher is explicitly Givens-aligned, but even there the native independent bank and rank-2 ordinary basis achieve numerical-zero error. The smaller Mirror payload has nonzero error and much higher eager CPU latency; unrelated adapters require private capacity.

## U — boundaries

This experiment cannot establish image quality or T2I performance. A positive result applies only to this fixed-feature composition mechanism screen.
