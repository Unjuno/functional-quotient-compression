# MA-598 — Hash-addressed logical experts

Status: **FAIL (development; fresh sealed)**. Prior art PA120 HashedNets and PA02 shared MoE routing. The experiment tests four specialist functions over a shared hashed FFN and charges all expert views, router state, and physical weights.

## H — Hypothesis

A shared 2,048-bucket HashedNet table plus per-expert Givens Views can provide four useful specialists at fewer actual bytes than independent experts, while improving over ordinary tying and per-expert native hash salts.

## T — Planned execution

Digits modulo four formed the four expert groups. Each expert trained on its group with class-masked logits; a shared linear router predicted the group. Compared independent dense experts, independent hash tables, tied shared hashing, salted shared hashing, Givens Mirror, diagonal gates, and rank-one residuals. Development seeds 59801/59802 completed; fresh seeds 59811–59813 remain sealed. Per-expert oracle quality and route use passed. Mirror failed the frozen +1pp-vs-controls and byte gates.

## D — Decision

**FAIL.** Both development seeds failed the complete Mirror gate. Fresh data remained sealed. See `RESULTS_CORE.csv`, `runs/dev_*/metrics.json`, and `VERIFICATION.json`.

## C — Strongest counter-hypothesis

Native expert-specific hash salts or ordinary gates may already create enough distinct functions; the router or shared output head may dominate measured quality.

## U — Boundaries

**Fact:** Across dev seeds 59801/59802, router-selected accuracy was: tied .9267/.9289, salted .9289/.9311, Mirror .9289/.9333, diagonal .9267/.9311, rank1 .9267/.9311, dense .9267/.9333. Mirror was only 0.00pp/+0.22pp over salted and +0.22pp/+.00pp over tied, missing the required +1pp. Mirror payload was 11,218 B in both seeds versus salted 10,735 B (+4.50%); Mirror was 13.9% of independent dense payload. Each route fraction was 19.1–31.3%, and each expert's oracle accuracy was >=.984 in all methods. Mirror's training wall was 1.92/2.09 s versus salted 1.34/1.44 s; measured inference throughput was lower (451k/402k vs 553k/610k examples/s), with short CPU timing windows.

**Metric limitation:** The serialized-output router-selected CE equals 68,888,880 and 60,000,000 for all methods because the code computes CE on logits where all non-specialist classes are set to -1e9. The CE metric is therefore non-discriminating and is marked NOT USABLE; no post-freeze metric substitution was made. Accuracy, oracle quality, routing, payload, and diagnostics remain usable under the frozen gates.

**Interpretation:** This screen shows useful conditional specialists can share a physical hash table at much lower payload than independent experts, but learned Givens Views do not add quality over native expert-specific hash addressing and cost more bytes and compute. Logical expert count alone is not evidence of independent capacity.

**Hypothesis:** View rotations align the tied hash functions enough to separate raw expert outputs (mean raw-logit cosine .979/.979, vs .249/.255 salted) but that separation is not useful for routed task quality; the native salts are already sufficient. The routing/specialist design and modest digits task may cap the measured benefit.

One small image dataset and four label groups do not establish broad MoE scaling or language-model capacity.
