# MA-1037 status

- Status: FAIL (one-model, one-benchmark retrieval screen)
- Branch: `research/ma-1037-matryoshka-dimension-mirror-20261115`
- Protocol and backend amendments frozen before test audit: yes
- Development: 809 train-qrels query IDs; five angles locked
- Fresh audit: 300 disjoint test query IDs complete
- Results/serialization replay: exact
- Registry/global summary: pending commit

Mirror's best nDCG@10 gain is +0.315 pp at 64 dimensions, below the +0.5 pp gate; no other sub-full dimension improves meaningfully. The normalized linear boundary control exactly aliases Mirror at every dimension. The final five-budget bank costs 230 B more than native. Runtime is NOT ESTABLISHED: repeated short-batch QPS results changed ordering; the final d512 Mirror rate is 19.0% below native. Encoder construction cost is 1,354.9 s for 5,183 corpus docs on CPU.

See `RESULTS.md`, amendment records, `runs/`, and `VERIFICATION.json`.
