# MA-371 status

- Status: **FAIL** (bounded nested-width MLP screen)
- Branch: `research/ma-371-matformer-granularity-mirror-20261008`
- Protocol frozen before development: yes; selected learning rate 0.003 on worlds 37100/37101
- Fresh worlds 37110/37111/37112: evaluated once at frozen setting
- Results: 105 rows across widths, methods, worlds and development rates
- Verification: 3 tests pass; payload hashes, inference roundtrip and metrics replay checked
- Next candidate: MA-372 (Mix'n'Match views), a separate hypothesis

## H — hypothesis

Per-layer, per-width four-angle Givens views improve nested shared-subnetwork quality by at least 1 point, stay within 2 points of independent networks, use at most half their complete payload, and beat same-budget FiLM/LoRA.

## T — execution

Two-layer nested MLP, width 64 and active widths 16/32/64; 700 sandwich-rule updates with distillation, 120 code updates per width. sklearn digits, stratified 60/20/20 worlds. Development seeds 37100/37101 selected LR 0.003 from {0.003,0.01}; fresh seeds 37110/37111/37112. Controls: nested baseline, Mirror, equal-code FiLM, rank-1 LoRA, independent models. PyTorch CPU, one thread. Actual complete torch.save banks measured.

## D — decision

**FAIL.** Fresh mean accuracy: nested 94.85%, Mirror 95.28%, FiLM 95.09%, LoRA 95.06%, independent 95.68%. Mirror's +0.43-point delta misses the frozen +1-point criterion. Mirror payload 40,063 bytes is 65.4% of independent payload 61,229 bytes, above the 50% limit. FiLM matched closely at equal bytes with lower correction MACs.

## C — strongest counter-hypothesis

Nested-prefix training already shares most useful features; any small specialization is ordinary feature conditioning, while the orthogonal view adds compute without enough quality or byte gain.

## U — unresolved

No Transformer-scale MatFormer reproduction, no unseen mixed-granularity configurations (MA-372), no near-convergence capacity evidence, and no device latency benchmark.

## Evidence separation

- **Fact:** 105 rows replay; payload hashes match; max metric replay difference 4.6e-9; serialization logits exactly match.
- **Interpretation:** small-width nested sharing is compact, but this Mirror insertion fails the preregistered frontier.
- **Hypothesis:** a factorized layer×granularity code for unseen mixes may behave differently; that is MA-372 and remains untested.
