# MA-599 — Factorized hash-bucket View codes

Status: **FAIL (development; fresh sealed)**. Prior art PA120 HashedNets. This tested a bucket-space low-rank basis with specialist coordinates, distinct from input Givens (MA-597) and expert-specific hash salts (MA-598).

## H — Hypothesis

A shared rank-4 bucket basis plus four rank-4 codes can produce useful conditional specialist functions from one shared HashedNet table, outperforming tied and salted hash controls at comparable actual bytes and surviving comparison with an ordinary dense rank-4 residual.

## T — Frozen execution

Use sklearn digits labels modulo four as class groups, four specialists and a shared linear router; each expert is trained only on its class group. Methods: independent dense, independent hashed tables, tied shared hash, salted shared hash, factorized hash-bucket View, diagonal gate, and dense rank-4 residual. Each method gets 800 AdamW updates, matched router initialization/batches, dev seeds 59901/59902. Fresh seeds 59911–59913 remain sealed unless the complete dev gate passes. The primary CE is hierarchical: router group NLL plus conditional class NLL within the true group. All inference payload objects are serialized to NPZ and charged.

## D — Decision

**FAIL.** Factorized View did not meet the quality threshold over tied/salted controls, exceeded the salted byte cap by 2.576x, and was matched by native rank-4 residual at a smaller payload on seed 59901. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Native salted addressing or an ordinary low-rank residual captures any useful expert separation; bucket-space factorization could simply behave as a shared low-rank parameter basis.

## U — Boundaries

## Facts / interpretation / hypothesis

- **Fact:** Seed 59901 factorized View accuracy .9133 vs salted .9111 (+.22pp), hierarchical CE .2477 vs .2527, bytes 27,651 vs 10,735; native rank-4 dense residual has the same .9133 accuracy at 17,337 B. Seed 59902 View and salted both have .9133 accuracy; hierarchical CE is .2635 vs .2591, and rank-4 residual has .9111 at 17,337 B. Four router experts each receive 19.1–31.6% of examples, and View per-expert oracle accuracy is >=.985. Each method got 800 updates. Fresh seeds remain sealed.
- **Interpretation:** Bucket-space factorization creates specialist outputs, but its learned shared basis consumes more bytes than the native low-rank residual and fails to provide a stable quality advantage over hash salts. Hash sharing alone remains more storage-efficient.
- **Hypothesis:** The large bucket-space basis pays for a generic 4D correction with poor storage amortization; most measured gain can be captured by ordinary low-rank residuals or native hash addressing.

This small digits specialist screen does not establish broad MoE scaling, language-model capacity, near-convergence capacity or optimized-kernel runtime. Logical expert count is not independent capacity.
