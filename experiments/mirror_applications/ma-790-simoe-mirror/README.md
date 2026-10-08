# MA-790 — Factorized Mirror codes for SIMoE interpolation coefficients

Status: SCREENING  
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Prior art: PA208 — Sparse Interpolated Mixture-of-Experts

**Mirror insertion:** this experiment adds a small factorized task coordinate `m` to SIMoE's sparse interpolation coefficient generator so multiple logical experts can share coefficient structure while retaining sparse anchor mixtures.

## H — falsifiable hypothesis

Two-dimensional per-factor Mirror coordinates can recover unseen factor-composition functions with lower actual payload than a free sparse SIMoE coefficient row, while matching unrestricted low-rank coefficient factorization in the aligned case and failing gracefully for off-orbit functions.

## T — frozen screen

Use 12 fixed linear anchor experts and 16 logical tasks arranged as a 4×4 factor grid. The task function is the sparse top-3 interpolation of anchor outputs. Compare the native free SIMoE sparse coefficient rows, factorized Mirror codes, an equal-rank unrestricted additive factorization, and a no-code shared function. Train on the even-parity task pairs; hold out the entire odd-parity pairs. Run both factor-aligned and independent off-orbit coefficient worlds with three seeds. Held-out methods may use their declared 64-example support adaptation; query data remains untouched.

The task is deliberately controlled and synthetic. It measures interpolation coefficient sharing and composition, not language-model quality or the full SIMoE training recipe. See `PROTOCOL.json` for the exact anchors, sparsity, objective, update budgets, metrics, and gates. Draw28 replay is in `source/draw28_exclusions.json`.

## D — decision

Pending.

## C — strongest counter-hypothesis

Sparse interpolation coefficients already are compact expert addresses. Any gain from factorization is generic low-rank parameter sharing and is matched by the ordinary low-rank control.

## U — boundaries

The fixed linear anchor bank and synthetic coefficient worlds are a mechanism screen, not a full neural MoE upcycling result. No claim about LLM loss, routing, throughput, or production deployment follows.
