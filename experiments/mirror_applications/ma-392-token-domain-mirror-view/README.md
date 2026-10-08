# MA-392 — Factorized token × domain Mirror embedding address

Status: FAIL (development screen; fresh sealed)  
Evidence lane: MECHANISM / STORAGE / UNSEEN COMBINATIONS / RUNTIME  
Base commit: `bcf903fa3055d60ca45e99beb66bf56a5ae00aeb`  
Prior art: PA59 (compositional / quotient-remainder embeddings)

## Hypothesis

H: Shared quotient/remainder partition tables plus one learned Givens coordinate per domain can recover domain-specific token embeddings on token-domain combinations withheld from training. The view must add held-out task quality beyond tied composition and simple FiLM/additive controls while remaining within the actual-byte and CPU gates.

## Mirror insertion

The physical q/r tables stay shared. For token `(q,r)` in domain `d`, the Mirror view rotates the two 8D partition vectors by learned angle `alpha[d]`. Domain codes, both partition tables, classifier, metadata, and serialization overhead are all charged.

## Controls

- Independent embedding for every token-domain pair.
- Shared q/r concatenation tied across domains.
- Domain FiLM scale and bias over the shared embedding.
- Domain-specific additive 16D vector.
- Mirror Givens angle per domain.

## Frozen split and gates

There are 64 tokens and 4 domains. A seed-locked 25% of the 256 token-domain pairs is held out, while each token and domain appears in other training pairs. Two development worlds use 65,536 examples and 512 optimizer updates per method. Fresh seeds remain sealed unless both development seeds pass every gate in `PROTOCOL.json`.

Actual deterministic ZIP/NPY FP16 payload bytes are authoritative. The score is quality on seen and held-out pairs, relative payload size, role embedding uniqueness, and serialized-state CPU throughput. This synthetic screen does not establish independent capacity or natural embedding quality.

## Results

Mirror produced all 256 distinct FP16 role embeddings and beat FiLM on held-out accuracy in both worlds. It still failed the frozen joint gate: one world missed the held-out margin, seen accuracy trailed FiLM, payload was 1.137x the smallest shared control, and CPU throughput was 0.241–0.334x tied concatenation. Fresh seeds remain sealed. See [REPORT.md](REPORT.md), [RESULTS_CORE.csv](RESULTS_CORE.csv), and [VERIFICATION.json](VERIFICATION.json).
