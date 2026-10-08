# MA-325 — Tensorized embedding domain Mirror views

Status: NOT ESTABLISHED after development task validation. Dedicated branch: `research/ma-325-tt-embedding-domain-mirror-20261008`.

## H — hypothesis

One shared TT embedding with a small domain phase may preserve two aligned domains with fewer actual bytes than direct shared-TT coefficients. An unrelated domain tests the need for private embedding state.

## Mirror insertion

PA34 motivates TT as a native embedding compression control. Mirror adds a domain phase to one selected TT core and charges both shared core directions. The embedding is tied to the output softmax head; each effective domain table is paid once and used for input lookup and logits.

## Frozen protocol

See `PROTOCOL.json`. Three synthetic 16-token bigram domains; two share a planted TT-core phase orbit and one is unrelated. TT shape 4x4x4x4, rank 4, 600 Adam updates. Fresh payload quality is measured after reload. This is a tiny language-model analogue, not a full nanoGPT or natural-text experiment.

## Development finding and disposition

**FACT:** Seeds 32501 and 32502 were run across all six frozen conditions. Test NLL stayed at approximately 2.773 nat/token (uniform baseline `ln(16)`), with accuracy about 0.06. This includes the independent full-table upper control. Diagnostics show the planted TT tables have small scale and the resulting transition probabilities are nearly uniform. A phase indexing error for the third (unrelated) domain was fixed before any fresh evaluation. Exploratory teacher rescaling variants did not make the frozen 600-update procedure learn the task and are excluded from confirmatory claims.

**INTERPRETATION:** The frozen protocol does not expose meaningful quality differences because the task is not learned, so it cannot test the storage/quality hypothesis.

**HYPOTHESIS:** No verdict on whether Mirror TT coordinates help domain embeddings.

**COUNTER-HYPOTHESIS:** Optimization failure in this synthetic tied-head setup, rather than the Mirror representation, explains the flat scores.

**UNKNOWN:** Whether a redesigned learnable teacher/task yields a Mirror benefit; all fresh seeds remain sealed.

Disposition: NOT ESTABLISHED. No status is assigned to the Mirror method itself; a corrected, separately versioned protocol is required.
