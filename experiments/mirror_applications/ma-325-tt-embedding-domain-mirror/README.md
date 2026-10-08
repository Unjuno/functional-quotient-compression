# MA-325 — Tensorized embedding domain Mirror views

Status: protocol frozen before development. Dedicated branch: `research/ma-325-tt-embedding-domain-mirror-20261008`.

## H — hypothesis

One shared TT embedding with a small domain phase may preserve two aligned domains with fewer actual bytes than direct shared-TT coefficients. An unrelated domain tests the need for private embedding state.

## Mirror insertion

PA34 motivates TT as a native embedding compression control. Mirror adds a domain phase to one selected TT core and charges both shared core directions. The embedding is tied to the output softmax head; each effective domain table is paid once and used for input lookup and logits.

## Frozen protocol

See `PROTOCOL.json`. Three synthetic 16-token bigram domains; two share a planted TT-core phase orbit and one is unrelated. TT shape 4x4x4x4, rank 4, 600 Adam updates. Fresh payload quality is measured after reload. This is a tiny language-model analogue, not a full nanoGPT or natural-text experiment.
