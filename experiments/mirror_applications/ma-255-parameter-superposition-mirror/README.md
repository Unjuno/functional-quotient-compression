# MA-255 — Parameter Superposition with structured Mirror contexts

Status: SCREENING. Evidence lane: storage / retrieval / interference.

## H / T / D / C / U

**H:** At equal serialized bytes, low-dimensional structured task contexts can improve held-out task retrieval over the fixed random Rademacher contexts used by Parameter Superposition.

**T:** Synthetic Gaussian teacher vectors (dimension 512, 32 tasks), held-out Gaussian inputs, with explicit vectors, original fixed PSP unbinding, shared tensor without task codes, and development-trained 8D Mirror contexts. Fresh worlds 25510–25512 × seeds 0–2. All inference state serialized and charged.

**D:** NOT ESTABLISHED (development stop before fresh). Fixed-context PSP unbinding yielded held-out NRMSE 1.000; rank-8 learned contexts yielded 1.151–1.154 at 19,237 B/task; shared sum without task code yielded 5.805. None met the preregistered absolute utility ceiling (<0.25), so fresh worlds were not opened and no comparative Mirror claim is made.

**C:** Any gain may come from fitting a small shared codebook to aligned synthetic task vectors, which a generic learned latent code can match; original PSP may already be optimal for incoherent random tasks.

**U:** A task regime where native PSP has useful retrieval; learned-context generalization; any neural model-composition benefit. The current synthetic independent-vector setup did not establish useful superposition.
