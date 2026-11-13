# MA-930 result — real-LM bridge screen

## H — hypothesis

A source × target × layer factorized transport `W(s,t,l)=B_t C_l A_s` can reconstruct useful cross-model block connections for unseen ordered model pairs with lower state bytes, while preserving target-side function behavior versus independent affine connectors.

## T — what ran

- Frozen amendment 01 on CPU with three public GPT-Neo TinyStories models: widths 64/128/256, 8 layers, shared 50,257-token vocabulary.
- Connector fit: 256 train stories, max length 32; validation: 128 separate stories, max length 32. Holdout directions were 1M→8M and 8M→1M. Four remaining directed pairs trained the factorized basis; independent affine controls were fit per pair/layer.
- Rank 8, bridge layers 2/4/6/8, 1,000 updates, batch 256, three bridge seeds 93001/93007/93013.
- Measured held-out hidden MSE, target-tail KL to native target logits, next-token NLL delta, connector state bytes, full deployment bytes (all three model snapshots plus connectors), connector MAC proxy, and bridge-training wall time.
- Checkpoint reload replay max difference was 0. A separate tail-path identity check gave max logit difference 0 at layers 2/4/6/8 when passing native target states through the target tail.
- Data/model file sizes and SHA-256 are recorded in `source/data_manifest.json`; temporary model/data assets remain outside Git.

## D — FAIL for this factorized bridge mechanism

### Fact

| Mean over 48 held-out pair × layer × seed rows | Factorized | Independent affine | Difference |
|---|---:|---:|---:|
| Hidden MSE | 3.607 | 1.348 | 2.68× higher |
| Target-tail KL | 8.770 | 4.495 | 1.95× higher |
| Next-token NLL delta vs native target | +8.933 | +4.572 | 1.95× higher |
| Connector bytes | 44,069 B | 1,864,027 B | 97.6% fewer |
| Full deployment bytes | 237,829,794 B | 239,649,752 B | 0.76% fewer |
| Connector MAC proxy per token | 2,784 | 16,544 | 83.2% fewer |
| Bridge training wall time | 1.995 s | 1.725 s | 15.6% slower |

Factorized NLL drift exceeded independent control on all three seeds and both held-out directions. The independent control itself also had substantial NLL drift; neither control establishes successful semantic stitching.

### Interpretation

The source × target × depth basis sharply compresses connector state and reduces its matrix-multiply proxy, but this configuration loses too much target behavior to meet the quality gate. Because the backbone weights dominate deployment, the 97.6% connector-state reduction yields only 0.76% fewer full deployment bytes. Factorized connector training was slower here despite the lower inference MAC proxy.

This is a scoped representation-transfer failure for rank-8 `B_t C_l A_s`, not a StitchLLM reproduction or semantic-equivalence conclusion. The factorization is an ordinary tensor-basis control and shows no Mirror-specific advantage.

### Hypothesis

A single shared low-rank operator basis cannot capture noncommuting, pair-specific representation transport across these model scales; a private residual per held-out interface may be necessary and could consume much of the byte savings.

## C — strongest counter-hypothesis

The TinyStories models are small scale variants, only 256 fitting stories were used, and independent affine stitching also had large NLL drift. The failure may reflect insufficient adapter fitting or this model/data family rather than a general limit on factorized connectors. PA241 cautions that representation matching alone does not establish shared information.

## U — unresolved

Native StitchLLM routing/throughput, broader and stronger pretrained LLM families, more connector training data and convergence, private residual allocation, downstream task quality, and whether a small residual recovers the held-out NLL without consuming the connector-byte savings.
