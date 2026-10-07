# MA-003 status

- Status: PROMISING (aligned synthetic expert-view mechanism; formal all-world quality gate missed once)
- Branch: `research/ma-003-mirror-topk-expert-20261007`
- Base commit: `3826b43b4f474e8739392322d399e50c1ac11e6b`
- Development complete: yes; final v3 selected LR 0.01
- Fresh/audit opened: yes; worlds 30001–30003
- Results committed: yes (`e50a20fe4c000ffb3113f9d3e6564b3efacd4395`)
- Verification committed: yes (`e50a20fe4c000ffb3113f9d3e6564b3efacd4395`)
- Registry row updated: yes (tracker commit pending)

## Decision

**H — Hypothesis.** Four top-1 logical expert views of one shared physical matrix recover a four-expert teacher generated from that same shared matrix plus per-role Givens views at lower actual serialized bytes than untied MoE. Independent role matrices should expose the need for private parameters.

**T — Test.** 16D-to-12D linear experts, four roles routed by the first two input signs, top-1; 1,200 AdamW updates at the development-selected LR 0.01; three fresh worlds (30001–30003); six methods (full MoE, low-rank-router full MoE, tied, scalar gate, rank-1 residual, Mirror); aligned-view and independent-expert teachers. Each method shares matched minibatches within a world/mode.

**D — Decision: PROMISING.** Aligned Mirror MSE was within the 1.10x full-MoE gate in 2/3 worlds (1.145x, 0.887x, 0.972x); route accuracy was within 1 percentage point in 3/3; 3,920B vs 6,102B serialized payload (35.8% fewer) in all worlds. It substantially beat tying, scalar gate, and rank-1 residual on aligned quality. It missed the strict 3/3 MSE gate once, so this is not a PASS. In independent mode, Mirror MSE 3.42–3.63 vs full MoE 0.19–0.29, demonstrating a task-specific private-parameter boundary.

**C — Strongest counter-hypothesis.** The result may primarily reflect a teacher constructed exactly from the Mirror coordinate family and a short fixed-budget regression screen. Longer training or a broader task distribution may change the relative ranking; the rank-1 control is not an exhaustive learned low-rank family.

**U — Unconfirmed.** Near-convergence/fixed-byte capacity, natural language quality, larger nonlinear FFN experts, top-k greater than one, optimized-kernel latency, and whether learned LoRA/FiLM can match the aligned teacher at equal payload remain untested.

## Evidence

### Fact

- The v3 freeze manifest matched all source/protocol hashes before fresh access.
- 36 fresh result rows were independently replayed; max MSE delta 4.8e-10, route-accuracy delta 5.0e-9, R² delta 4.9e-9, payload-byte delta 0.
- Tests: 4 passed.
- Aligned Mirror active-compute proxy was 73.7M vs full MoE 191.7M per training example budget; median measured training time was 1.23s vs 1.21s. Median inference throughput was 0.85M vs 1.17M examples/s on this single-thread CPU.
- Independent-mode Mirror remained far worse than full MoE despite lower payload and compute proxy.

### Interpretation

The aligned mechanism produces useful logical multiplicity from one shared expert and reduces payload, but the observed quality gate is not robust enough for a PASS. CPU measurements show a runtime cost despite lower active-compute proxy. The independent condition shows that coordinate views cannot encode arbitrary per-role matrices.

### Hypothesis

A larger nonlinear MoE screen should test whether the aligned storage-quality tradeoff persists after training approaches convergence, and whether a learned low-rank or FiLM control can reach the same frontier.
