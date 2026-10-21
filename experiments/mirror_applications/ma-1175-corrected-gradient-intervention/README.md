# MA-1175: Corrected Mirror supergraph intervention experiment

**Mirror insertion:** Same MA-1174 128-scalar shared core produces 32 virtual 8→8→8 MLP paths via learned addresses and Householder–tanh–inverse views. Structure intervention aims to improve later shared-core learning, not simply remove low-weight paths.

**Changes:** exact per-path shared-core VJP (not gate sign disagreement), no-cut candidate with explicit positive dev margin, short H=6 vs long H=40 dev lookahead, H=120 dev-only after-the-fact audit, independent new teacher seeds, compute-matched keep extra, final-mask-from-start, identical-address non-Mirror control. Do not change historical MA-1174 verdict. No trained natural language model.

Frozen protocol SHA-256 `e00d9848d3948ab52d52ddc7240e0dafefd513c64102d969d272295107a3f790`; full JSON in `PROTOCOL.json` (committed before training).