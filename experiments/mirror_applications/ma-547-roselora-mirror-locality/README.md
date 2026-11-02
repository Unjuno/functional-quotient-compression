# MA-547 — Sparse weight edit versus routed hidden-space edit locality

Status: FAIL
Branch: `research/ma-547-roselora-mirror-locality-20261009`
Base commit: `52a11e2f79b626865a78274bbbd5dd682d428c00`
Prior art: PA96 (ReFT/LoReFT), PA104 (RoseLoRA)

## H — Hypothesis

A key-routed residual-space edit vector can transfer a synthetic counterfactual value to held-out paraphrases with better efficacy/locality than sparse RoseLoRA at similar actual bytes, and may justify its 512D state over a gated scalar-logit control.

## T — Execution

Pinned Pythia-70M-deduped, final normalized hidden state, 16 synthetic key/value relabelings, four support and four held-out paraphrases per key. Development seeds 54701/54702. The hidden intervention is an analytic 512D vector that raises the target-over-old output-row margin by 2.0. Controls are an ungated sparse rank-one output-row RoseLoRA update, the same row update with the same exact UTF-8 key gate, and a gated +2.0 scalar logit bias. Support/query split and every inference NPZ are retained.

Amendment 1 replaced a failing BPE-subsequence router with exact key-string matching; the initial run is preserved in `results/pre_amendment_1/`. Corrected dev seed 54701 replay is exact. Fresh seeds 54711–54713 remain sealed because one development seed missed the fixed edit-efficacy gate and the gated scalar control dominated bytes at matched edit/locality.

## D — FAIL

The Mirror vector misses the edit-success gate in development seed 54701 (0.641 vs required 0.75). In both development seeds, the gated scalar-bias control matches Mirror edit success, margin gain and zero non-target KL at 3,026 B versus Mirror's 35,990 B. Same-gated RoseLoRA also matches efficacy/locality within 8 B. Therefore no fresh worlds were opened.

## Facts

- Exact key routing accuracy is 1.00 in both corrected development worlds.
- Mirror edit success is 0.6406 / 0.7500; mean target-vs-old margin gain is 2.0000 in both.
- Gated bias has the exact same edit success, +2.0 margin and zero non-target KL at 3,026 B. Same-gated RoseLoRA uses 35,982 B vs Mirror 35,990 B and has effectively identical edit success/locality.
- Ungated RoseLoRA gives the same edit success but changes non-target distributions: mean KL 3.44e-5 / 2.23e-5; max KL 5.92e-4 / 7.13e-4. Non-target top-1 retention remains 1.00, with mean absolute target-vs-old margin shift about 2.0.
- The hidden vector's actual payload is 35,990 B versus 3,026 B for scalar bias. Pythia standalone model bytes are 168,144,624 B, dominating full-system storage.
- The synthetic answer tokens are random ordinary vocabulary IDs; this is not a natural factual-edit benchmark.

## C — Strongest counter-hypothesis

The exact key gate already provides locality. The desired edit is only a scalar change to one answer logit, so a gated bias implements it directly and uses about 11.9x fewer incremental bytes. A residual vector is a ReFT-style way to implement the same scalar response, not a new Mirror-specific function.

## U — Not established

Fresh-world transfer is untested because the dev efficacy gate failed. Natural factual edits, fuzzy aliases, multi-token answers, longer generations, and other insertion layers remain untested.

## Fact / Interpretation / Hypothesis

Fact: A +2 margin edit vector has 0.641–0.750 held-out success, while the gated scalar control matches it at lower bytes. Interpretation: this output-interface screen provides no Mirror-specific locality or storage value. Hypothesis: activation views may earn value when edits need distributed, multi-token behavior that scalar output changes cannot express; a separate protocol would need to test that.
