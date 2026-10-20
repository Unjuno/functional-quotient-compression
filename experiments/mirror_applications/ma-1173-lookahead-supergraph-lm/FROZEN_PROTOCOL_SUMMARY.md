# MA-1173 — Frozen prospective cut-to-learn experiment (2026-10-21 JST)
**Protocol hash:** `bdfaf85229434d6d3e9df3f0f33e2f640bf65c0cfe8059bde676307e1847da49` (full local PROTOCOL.json, frozen before any training; sync the full version when source is uploaded).

**Mirror insertion:** one 288-scalar physical 2×12×12 core is mapped through four learned address pages, per-edge gates and Householder→tanh→Householder to 18 logical 12→12→12 MLP edges, fully connected between 3 state nodes with 2 directions and 2 repeated loops. A separate causal 2-head attention gives token mixing. Every path has a hard ON/OFF mask separate from its learned weights. Only paths (not node count, loop count, attention, or view count) are subject to hard removal in this first test.

**H:** Selecting edge removals by prospective short training rather than current gate magnitude can improve *future learned rules* in a tiny causal language task. Compare mirror_keep, mirror_random, mirror_low_gate, mirror_lookahead, mirror_guarded, direct_tanh_lookahead, affine_lookahead, native_tied, mirror_keep_extra. Generating logical weights is not equivalent to independently learned functions.

**T:** CPU PyTorch FP32, 1 thread, seeds 31/47/59 × balanced/shortcut key→value grammar (not natural text). Train N=4096, dev=512, fresh=1024, OOD=1024; new data seeds per split; next-token CE + 2× final query token CE. AdamW lr .005 batch32 weight decay 1e-5 grad clip1, 260 real updates, cuts at 80/140/200 of 2 edges each. Each policy considers the same four deterministic candidate edge pairs per event. Lookahead uses cloned model+optimizer, 8 same future train batches per candidate plus keep, *all 40 probe updates per event counted*. Guard accepts only an improvement over keep on dev of ≥0.0005 CE. Extra-training no-cut control receives 380 updates. Fresh never selects anything.

**D:** H_learning requires ≥0.02 absolute OOD final-token accuracy gain versus BOTH mirror_random and mirror_low_gate in ≥4/6 paired cases, while fresh-ID mean token NLL no worse than 1.01×. H_Mirror requires superiority versus BOTH direct_tanh and affine at comparable total real storage/compute. Tests: causal leakage, edge count/real skipped computation, physical weight and gate gradients, serialized packet bytes and checkpoint replay. Any non-confirmation records FAIL, not capacity proof.

**C:** improvements could be due to generic path mask search, extra probe compute, shared-weight recurrence, or direct non-Mirror nonlinear maps. Strong native tied FFN at 1876 vs Mirror 1878 learned coefficients.

**U:** short synthetic grammar, 3 seeds×2 worlds, clock unlocked, no GPU/natural-language metric, no full structural plasticity or optimizer theorem. Report quality/bytes/active compute/CPU latency and both actual and probe updates.

Branch: `research/ma-1173-lookahead-supergraph-lm-20261021`. Do not merge into main. Do not edit historical verdicts or third-party nanoGPT.