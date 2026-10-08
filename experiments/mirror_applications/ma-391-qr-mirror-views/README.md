# MA-391 — Quotient-remainder Mirror compositional embeddings

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / UNSEEN COMBINATIONS / RUNTIME  
Base commit: `b444c99` (MA-389 result branch)  
Prior art: PA59 (quotient-remainder compositional embeddings)

## Hypothesis

H: A factorized angular View `m(q,r)=α_q+β_r` over quotient/remainder partition tables generalizes to held-out q×r token combinations better than fixed addition, elementwise multiplication, or concatenation while using at most 10% more serialized bytes than the best fixed composition and substantially fewer bytes than independent token embeddings.

## Mirror insertion

> **Mirror insertion:** this experiment adds the factorized angle `m(q,r)=α_q+β_r` to the quotient/remainder composition interface so that a logical token embedding can rotate the two partition vectors without storing a private embedding for each token pair.

- Native method: compose quotient and remainder table entries by addition, multiplication, or concatenation.
- Mirror method: apply one Givens rotation parameterized by `α_q+β_r` to the concatenated partition vectors.
- Codes `α_q` and `β_r` are learned partition-level state and are fully charged.
- Controls: independent table, all three native fixed compositions, and an unconstrained per-pair angle diagnostic on seen pairs.

## Physical-to-logical claim

Two small complementary partition tables represent 144 q×r logical token embeddings. A locked subset of pair combinations is held out from training. The synthetic teacher is generated using a factorized Givens angle, which makes a positive result an aligned recombination test. The number of q×r addresses is not treated as independent capacity.

## Comparisons

1. Independent embedding table (unseen token IDs receive no learned vector).
2. Quotient/remainder addition.
3. Quotient/remainder elementwise multiplication.
4. Quotient/remainder concatenation.
5. Factorized Mirror rotation with `α_q+β_r`.
6. Per-pair free-angle diagnostic for trained/seen combinations; not a held-out generalization control.

All methods use the same train pairs, class labels, batches, updates, and classifier output dimension. Held-out combinations are locked before training.

## Gates

### PASS (development screen)

Both worlds: independent seen-pair accuracy at least 0.80; Mirror within 2 points of the best fixed native composition on seen pairs; Mirror improves held-out-pair accuracy by at least 5 points over concatenation; Mirror payload no more than 1.10x the best fixed-composition payload and no more than 0.50x the independent table; embedding uniqueness is measured over all q×r pairs; serialized-state CPU throughput at least 0.90x concatenation. Fresh runs only if every development condition passes.

### FAIL

Either development world misses a registered held-out, quality, storage, uniqueness, or runtime condition. Fresh remains sealed.

### NOT ESTABLISHED

The held-out combination split leaks into training, or serialized replay / uniqueness measurements fail.

## Tuning boundary

- Development worlds: seeds 39101 and 39102.
- Fresh worlds: seeds 39111, 39112, and 39113; sealed unless both development worlds pass every gate.
- No per-held-out-pair private code, label, or embedding is supplied at evaluation.

## Storage and compute

Actual deterministic ZIP/NPY FP16 inference payload bytes are authoritative. Count both partition tables, all `α` and `β` codes, classifier weights, hash/partition metadata, and archive headers. Report seen and held-out NLL/accuracy, unique embedding count, examples/updates, MAC/lookup proxy, fit time, and serialized-state CPU lookup throughput.

## Scope

This is a synthetic held-out recombination screen over a fixed vocabulary. It is not natural language or a large-vocabulary deployment result.
