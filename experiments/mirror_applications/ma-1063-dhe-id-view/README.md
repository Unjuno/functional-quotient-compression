# MA-1063 — DHE plus structured field Mirror codes

This experiment tests the exact PA338 insertion point: deterministic dense hash encodings feed a learned table-free DHE decoder, then user/item field coordinates transform the generated embeddings. Compare shared DHE with no view, paired diagonal gains, FiLM, Givens Mirror, full tables and independent DHE decoders.

The scope is an in-house CPU screen on MovieLens 1M binary explicit preference data. The DHE hash length is scaled from the paper's k=1024 to k=128, so this is not an official reproduction. MovieLens terms and citation are retained in DATASET_TERMS.md; the raw archive stays external. Fresh audit pairs and seeds are sealed in FRESH_FREEZE.json before numeric audit labels are used.
