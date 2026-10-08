# MA-372 status

- Status: **FAIL** (bounded mixed-width MLP screen)
- Branch: `research/ma-372-matformer-mixmatch-mirror-20261008`
- Protocol frozen before development: yes; selected LR 0.003 on worlds 37200/37201
- Fresh worlds 37210/37211/37212: evaluated once at frozen setting
- Results: 672 rows across 24 mixed configs, methods, worlds and development LRs
- Verification: 3 tests pass; payload hashes, exact output roundtrip and metric replay checked
- Next candidate: MA-374 (ALBERT shared layers + depth Mirror)

## H — hypothesis

Layer×width-factorized Givens codes learned only from homogeneous-width subnetworks compose into useful unseen mixed-width configurations, beating native nested execution and byte-near FiLM while remaining compact relative to a direct code bank.

## T — execution

Three-layer width-64 nested MLP; widths 16/32/64; 800 sandwich/distillation updates and 100 code updates per factor. Digits v1.8.0, stratified per-world 60/20/20. Development worlds 37200/37201 selected LR 0.003; fresh worlds 37210/37211/37212. Factorized Mirror and FiLM codes were trained only on the three homogeneous configurations; all 24 mixed triples were held out from code fitting. Direct Mirror control fit per mixed configuration. Full serialized banks measured.

## D — decision

**FAIL.** Fresh mean accuracy over 24 mixed configs and three worlds: nested 94.41%, factorized Mirror 94.56%, FiLM 94.70%, direct per-config Mirror 94.70%. Mirror gains only 0.16pp over native nested, below the 1pp gate. Factor code bank uses 57,968 bytes versus 74,803 bytes direct (77.5%, failing <=60%); FiLM costs the same as factorized Mirror and is more accurate.

## C — strongest counter-hypothesis

The nested model's width assignments already share a common representation; any incremental adaptation comes from ordinary feature scaling or per-config fitting, while the orthogonal code is not a useful unique compositional mechanism.

## U — unresolved

No Transformer MatFormer reproduction, no tasks beyond digits, no independent capacity inference from the 24 combinations, and no device-specific latency.

## Evidence separation

- **Fact:** 672 rows replay with max metric difference 4.9e-9; serialized payload hashes match and diagnostic logits round-trip exactly.
- **Interpretation:** factorized coordinates compose across widths but do not improve the tested quality/storage frontier over simple FiLM.
- **Hypothesis:** a shared code basis with a learned sparse residual might reduce factor-bank bytes; this requires a new preregistered experiment.
