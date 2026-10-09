# MA-598 — Hash-addressed logical experts

Status: **FROZEN_BEFORE_DEVELOPMENT**. Prior art PA120 HashedNets and PA02 shared MoE routing. The experiment tests four specialist functions over a shared hashed FFN and charges all expert views, router state, and physical weights.

## H — Hypothesis

A shared 2,048-bucket HashedNet table plus per-expert Givens Views can provide four useful specialists at fewer actual bytes than independent experts, while improving over ordinary tying and per-expert native hash salts.

## T — Planned execution

Use digits modulo four as four expert groups. Each expert trains on its group, with class-masked logits; a shared linear router predicts the group. Compare independent dense experts, independent hash tables, tied shared hashing, salted shared hashing, Givens Mirror, diagonal gates, and rank-one residuals. Development seeds 59801/59802; fresh seeds 59811–59813 stay sealed. Measure router-selected and oracle-expert quality, per-expert use, collision/diversity, compute, and actual serialized payload bytes.

## D — Decision

Pending frozen development results.

## C — Strongest counter-hypothesis

Native expert-specific hash salts or ordinary gates may already create enough distinct functions; the router or shared output head may dominate measured quality.

## U — Boundaries

One small image dataset and four label groups do not establish broad MoE scaling or language-model capacity.
