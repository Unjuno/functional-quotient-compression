# MA-598 status

- Status: **FAIL (development, verified)**
- Branch: `research/ma-598-hash-addressed-logical-experts-20261009`
- Development seeds 59801/59802 complete; fresh seeds 59811–59813 sealed.
- Tests: 3 passed. Deterministic replay: 14/14 method payloads byte-exact.

## H / T / D / C / U

- **H:** A shared hash table with small per-expert Givens coordinates would recover useful expert behavior, improving over tied and per-expert salted hashes at comparable actual inference bytes.
- **T:** Digits groups `label % 4`; 64→128→10 FFN specialists plus shared linear router; seven controls; 800 AdamW updates; development splits/seeds 59801 and 59802. Fresh seeds 59811–59813 were not opened. Actual NPZ payloads were serialized and loaded for evaluation.
- **D:** **FAIL.** Mirror misses the required +1 percentage point vs tied and salted controls on both seeds. The ≤1.05× salted byte cap passes, but Mirror uses 11,218 B vs 10,735 B (+4.50%). All four experts are used (19.1–31.3%) and every per-expert oracle accuracy is at least .984. Mirror training is slower than salt controls in both seeds. Test suite: 3 passed; 14/14 same-seed method payloads replayed byte-exactly.
- **C:** Native expert-specific hash salts already provide distinct useful specialists; simple diagonal/rank-one controls match Mirror quality at byte-near costs. The small digits/router setup may cap visible quality separation.
- **U:** Cross-entropy on the router-selected class-masked logits is unusable: every method receives identical huge CE because non-specialist logits are -1e9. No clean-task CE comparison, near-convergence capacity, larger MoE, GPU kernel, or language-model result is established.

## Facts / interpretation / hypothesis

- **Fact:** Seed 59801 Mirror/salted accuracy .9289/.9289; seed 59802 .9333/.9311. Tied accuracy .9267/.9289. Mirror payload 11,218 B; salted 10,735 B; dense 80,664 B. Fresh stayed sealed.
- **Interpretation:** Physical parameter sharing compresses this four-specialist screen substantially relative to independent dense experts, but the learned Givens View adds no demonstrated value over native hash salts and increases payload/compute.
- **Hypothesis:** Rotation changes raw expert functions but not their routed task utility; native hashing already supplies sufficient specialization for this task.
