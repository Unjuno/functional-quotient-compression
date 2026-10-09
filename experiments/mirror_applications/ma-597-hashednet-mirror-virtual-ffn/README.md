# MA-597 — HashedNet Mirror virtual FFN

Status: **FROZEN_BEFORE_DEVELOPMENT**. Prior art PA120. The protocol tests whether a small learned Givens input coordinate changes the quality/storage frontier of a hash-compressed FFN.

## H — Hypothesis

Thirty-two Givens angles can make a 2,048-bucket HashedNet's virtual FFN less sensitive to collisions, improving held-out digit accuracy over ordinary hashing at near-equal actual bytes and beating diagonal and rank-one controls.

## T — Planned execution

Train 64→128→10 MLPs on stratified scikit-learn digits splits with 800 AdamW updates. Compare dense first-layer weights, 2,048/4,096-bucket native HashedNets, 2,048 buckets plus Mirror Givens, diagonal gate, and rank-one residual. Development seeds 59701/59702; fresh seeds 59711–59713 are locked. Payloads charge all learned tensors, hash seed, metadata, and reconstruction state. Collision counts, teacher projection RMSE, actual bytes, inference compute, and wall time are reported.

## D — Decision

Pending frozen execution.

## C — Strongest counter-hypothesis

The observed benefit may be ordinary extra conditioning, diagonal input scaling, a low-rank residual, or the hash bucket count itself; a learned rotation may add compute without reducing functional collision loss.

## U — Boundaries

One small handwritten-digit dataset and one FFN layer do not establish language-model capacity or natural neural-network compression.
