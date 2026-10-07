# MA-248 status

- Status: **FAIL (Mirror-specific frontier not established)**
- Branch: `research/ma-248-packet-mirror-code-20261007`
- Base commit: `f91625f2fe1b110593b16605c26fc9c7675c1824`
- Development: complete; final v3 matched-minibatch screen selected LR 0.01
- Fresh/audit: complete; worlds 24801–24803
- Results committed: yes (`433a2229362db23bcd43b3830358e674799c6247`)
- Verification committed: yes (`433a2229362db23bcd43b3830358e674799c6247`)
- Registry row updated: see status board and registry

## H / T / D / C / U

- **H:** A shared packet code with phase-specific Givens views can preserve a jointly consistent four-token packet when source branch entropy is shared, and must beat simple shared-code controls to count as Mirror-specific.
- **T:** Six methods, two binary branch-source modes, two-block phase-slot decoder, 1,200 AdamW updates, final v3 minibatches matched across controls; dev world 24800 selected LR 0.01; fresh worlds 24801–24803. Actual serialized payload, address bits, training time, compute proxy, and CPU throughput were measured.
- **D:** FAIL for the registered Mirror-specific frontier. Mirror reached >=99% fresh joint accuracy in 2/3 correlated worlds, while broadcast code did so in 3/3 at 378 fewer serialized bytes. Mirror used no fewer address bits than PTP. Independent-mode quality was weak across controls and is NOT ESTABLISHED as an entropy boundary.
- **C:** The correlated Mirror miss may be fixed-budget optimization/initialization variance; the untied upper control also underperformed in one fresh world.
- **U:** Natural language, longer packets, continuous PTP auxiliaries, near-convergence/fixed-byte capacity, stronger optimization, and optimized kernels.

## Verification

Four tests passed. All 36 fresh metric rows replayed: max NLL delta 4.62e-10, max secondary metric delta 5e-9, exact payload-byte match 36/36. No fresh data was used for tuning.
