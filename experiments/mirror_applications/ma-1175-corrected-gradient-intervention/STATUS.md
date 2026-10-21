# MA-1175 Status

2026-10-21 JST — COMPLETE, **FAIL as to prospective learning benefit and Mirror specificity; mechanisms PASS**.

- Preregistration SHA256 `e00d9848d3948ab52d52ddc7240e0dafefd513c64102d969d272295107a3f790` was frozen on the research branch before 100 training runs.
- 100/100 models trained, 100/100 checkpoint hashes verified before fresh was opened, 100/100 dev checkpoint replays and generated-graph packet replays PASS. Source tests 10/10 PASS.
- Exact edge-to-shared-core VJP reconstruction max rel error `1.6666e-7`. Long lookahead hard cut accepted only 5/20 times with rule 20/20 valid.
- Primary predeclared learning improvement **0/10**; Mirror-specific superiority versus non-Mirror direct tanh **2/10** (threshold 7/10). Same final mask from initial training is no worse in 10/10 paired trials; extra training wins all 10.
- New result is not a natural-language LM experiment. Prior MA-1174 negative report stays unchanged. No main merge.
