# MA-470 — Shared Mirror edit basis with gradient-to-code generation

Status: **PROMISING (synthetic aligned edit-bank storage only)**
Evidence lane: edit basis / coefficient generation / locality / bytes
Protocol freeze: `d3edd370`; fresh worlds 47010–47012, seeds 0–2.
Base: `ae59be94`.

## H — Hypothesis

A shared rank-one edit atom with small Givens View coordinates and sparse per-edit coefficients can encode many rank-two edits more compactly than per-edit MEND factors. A private atom should restore edits outside the shared View orbit.

## T — Test

An 8×8 analytic linear editor used four rank-one atoms, with three Givens-related atoms and one independent atom. Each of 64 edits composed two atoms. We compared per-edit rank-two SVD factors (MEND-style), a generic independent four-atom bank, Mirror Views alone, Mirror plus a charged private atom, and a single shared atom. The coefficient code was least-squares projected from known edit matrices, an idealized upper bound rather than a learned MEND editor. Payloads were serialized with `torch.save`; all matrices, View angles, codes, indices and metadata were charged. Metrics cover edit NRMSE, locality drift, serialized bytes and measured payload-generation wall time.

## D — PROMISING, narrowly scoped

**Facts:** At N=64, Mirror plus private atom had 0.00000020 mean edit NRMSE (maximum 0.00000027), 3,793 actual bytes, and maximum locality drift 1.1e-7 across nine fresh world/seed pairs. MEND-style factors had zero reported edit error and 54,847 bytes; the generic independent atom bank also reconstructed edits at 3,165 bytes. Mirror Views alone used 3,289 bytes but mean error was 0.324 (maximum 0.389), exposing the off-orbit boundary. Single atom mean error was 0.599. Mirror+private payload generation averaged 3.49 microseconds/edit in this microbenchmark; MEND was 33.0 microseconds/edit. Projection/apply MAC proxies are specified in RESULTS_CORE.csv.

**Interpretation:** In this deliberately aligned analytic setup, one physical atom plus View angles and one private atom captured the edit bank with 93.1% fewer bytes than MEND factors. The generic four-atom control was 628 bytes smaller than Mirror+private and equally accurate, so Mirror did not win the strongest simple-control storage comparison. This is evidence for a shared-basis storage frontier versus per-edit factors, not a Mirror-specific advantage.

## C — Strongest counter-hypothesis

The benefit comes from a known low-dimensional shared edit basis and idealized oracle projection; a generic independent basis already reconstructs exactly with fewer bytes. The analytic task grants the encoder exact edit matrices and says nothing about learned gradient-to-code behavior on a neural model.

## U — Unknown

Learned MEND coefficient generation, nonlinear model edits, factual efficacy/paraphrase quality, sustained sequential interference, large-model locality, and deployment runtime are untested. Mirror-only codes fail for independent/off-orbit atoms; private state is required.

## Decision

**FACT:** Fresh results are stable across three worlds and three seeds. Mirror+private is far smaller than per-edit rank-two factors, but larger than the generic atom bank.
**INTERPRETATION:** A reusable basis can amortize aligned edit state; the best representation here is generic shared atoms, not the tested Mirror encoding.
**HYPOTHESIS:** A learned editor may discover compressible View-aligned edits, but this screen does not establish that claim.
**BOUNDARY:** Synthetic linear editing with oracle least-squares coefficients; no claim about neural MEND or factual editing.
