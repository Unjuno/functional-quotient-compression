# MA-790 — Factorized Mirror codes for SIMoE interpolation coefficients

Status: FAIL
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Prior art: PA208 — Sparse Interpolated Mixture-of-Experts

**Mirror insertion:** this experiment adds a small factorized task coordinate `m` to SIMoE's sparse interpolation coefficient generator so multiple logical experts can share coefficient structure while retaining sparse anchor mixtures.

## H — falsifiable hypothesis

Two-dimensional per-factor Mirror coordinates can recover unseen factor-composition functions with lower actual payload than a free sparse SIMoE coefficient row, while matching unrestricted low-rank coefficient factorization in the aligned case and failing gracefully for off-orbit functions.

## T — frozen screen

Use 12 fixed linear anchor experts and 16 logical tasks arranged as a 4×4 factor grid. The task function is the sparse top-3 interpolation of anchor outputs. Compare the native free SIMoE sparse coefficient rows, factorized Mirror codes, an equal-rank unrestricted additive factorization, and a no-code shared function. Train on the even-parity task pairs; hold out the entire odd-parity pairs. Run both factor-aligned and independent off-orbit coefficient worlds with three seeds. Held-out methods may use their declared 64-example support adaptation; query data remains untouched.

The task is deliberately controlled and synthetic. It measures interpolation coefficient sharing and composition, not language-model quality or the full SIMoE training recipe. See `PROTOCOL.json` for the exact anchors, sparsity, objective, update budgets, metrics, and gates. Draw28 replay is in `source/draw28_exclusions.json`.

## D — decision

**FAIL for the preregistered held-out composition hypothesis.** Across 3 worlds × 3 initialization seeds × 8 odd-parity tasks, aligned zero-shot Mirror mean query MSE was 0.2505, versus 0.2178 for ordinary low-rank and 0.000064 for native SIMoE after 300 support updates. Mirror exceeded the 1.05× low-rank gate in all three worlds (world means 0.1957/0.3903/0.1655 vs 0.0939/0.4341/0.1255; the middle world passes but only 1/3, not the required 2/3), and did not meet the adapted-native gate. Its factorized state was 720 B versus the 928 B free 16-row coefficient reference (77.6%), so the storage gate passed. Off-orbit Mirror mean MSE was 0.6967; ordinary low-rank 0.7030 and adapted native 0.000096. This does not establish that Mirror coordinates improve SIMoE: the aligned quality gate failed and the simpler low-rank control is smaller (680 B factor/code state vs Mirror's 720 B) and has lower pooled aligned error.

The complete 1,008-row audit CSV retains all task/seed results. Mean recorded training wall time sums to 1,098.6 seconds across the audit reruns; CPU batch-1 coefficient-generation latency averaged 0.074 ms for Mirror, 0.039 ms for ordinary low-rank and 0.024 ms for native/no-code. These timings are CPU mechanism measurements, not end-to-end SIMoE throughput. Actual payloads are measured directly from serialized safetensors; full Mirror payload is 7,424 B (anchors, generator and codes) and the free-row reference is 7,632 B (including the same anchors).

## C — strongest counter-hypothesis

Sparse interpolation coefficients already are compact expert addresses. The Mirror polar coordinates did not generalize held-out factor pairs robustly and add coordinate-generation work; ordinary low-rank is both simpler and smaller in this screen.

## U — boundaries

The fixed linear anchor bank and synthetic coefficient worlds are a mechanism screen, not a full neural MoE upcycling result. No claim about LLM loss, routing, throughput, or production deployment follows. Mirror was zero-shot while native SIMoE was permitted 64-shot support adaptation by protocol; the latter is therefore a useful adaptation reference rather than an equal-inference protocol comparison.
