# MA-325 — Tensorized embedding domain Mirror views

Status: NOT ESTABLISHED after development task validation. Dedicated branch: `research/ma-325-tt-embedding-domain-mirror-20261008`.

## H — hypothesis

One shared TT embedding with a small domain phase may preserve two aligned domains with fewer actual bytes than direct shared-TT coefficients. An unrelated domain tests the need for private embedding state.

## Mirror insertion

PA34 motivates TT as a native embedding compression control. Mirror adds a domain phase to one selected TT core and charges both shared core directions. The embedding is tied to the output softmax head; each effective domain table is paid once and used for input lookup and logits.

## Frozen protocol

See `PROTOCOL.json`. Three synthetic 16-token bigram domains; two share a planted TT-core phase orbit and one is unrelated. TT shape 4x4x4x4, rank 4, 600 Adam updates. Fresh payload quality is measured after reload. This is a tiny language-model analogue, not a full nanoGPT or natural-text experiment.

## A3 amendment and development finding

**FACT:** Seeds 32501 and 32502 were run across all six frozen conditions. Test NLL stayed at approximately 2.773 nat/token (uniform baseline `ln(16)`), with accuracy about 0.06. This includes the independent full-table upper control. Diagnostics show the planted TT tables have small scale and the resulting transition probabilities are nearly uniform. A phase indexing error for the third (unrelated) domain was fixed before any fresh evaluation. Exploratory teacher rescaling variants did not make the frozen 600-update procedure learn the task and are excluded from confirmatory claims.

**INTERPRETATION:** Original A0/A1 task did not expose meaningful quality differences. A3 raises teacher logit scale based only on development validation; its confirmatory development rerun will determine whether fresh evaluation is permitted.

**HYPOTHESIS:** No verdict on whether Mirror TT coordinates help domain embeddings.

**COUNTER-HYPOTHESIS:** Optimization failure in this synthetic tied-head setup, rather than the Mirror representation, explains the flat scores.

**UNKNOWN:** Whether a redesigned learnable teacher/task yields a Mirror benefit; all fresh seeds remain sealed.

Disposition: NOT ESTABLISHED. No status is assigned to the Mirror method itself; a corrected, separately versioned protocol is required.

## A3 amendment

The original scale 8 task was not learned even by independent full. A validation-only development sweep on seeds 32501/32502 compared scales 8/12/16/24/32; scale 32 was selected (independent-full development validation macro NLL ~1.16, aligned accuracy ~0.67). Only teacher logit scale changes. Architecture, update budget, optimizer, LR, data volume, controls, gates, and fresh IDs remain fixed. Fresh IDs 32511-32513 remain unopened until the A3 confirmatory development gate is checked. The sweep used validation NLL only; no test/fresh metric informed selection.


## A3 confirmatory development result

**H:** on a learnable tied-head task, the domain phase should preserve aligned-domain quality and save at least 15% actual bytes over direct coefficients; an unrelated domain tests the private-state boundary.

**T:** Only the teacher logit scale changed to 32 after development-validation-only diagnostics; both development seeds (32501/32502), six controls, 600 Adam updates, LR 0.01, and data volumes were held fixed. The source and protocol for the original scale-8 run are retained as `source/run_A0.py` and `PROTOCOL_A0.json`. Fresh IDs 32511-32513 were not opened.

**D: FAIL for the frozen Mirror-specific byte gate (fresh remains sealed).** Independent-full validation macro NLL was 1.156 and 1.259, with macro accuracy 0.613 and 0.576, so the task is learnable under both development seeds. Aligned Mirror validation NLL was no worse than independent TT by the declared 0.05 nat margin. However, actual Mirror payload was 2,631B vs direct coefficients 2,637B (ratio 0.9977), far above the maximum allowed 0.85. Because the frozen development gate failed, fresh seeds remained sealed.

**C:** ordinary direct coefficients reproduce essentially the same quality and bytes; the observed 6B difference is not a useful Mirror-specific Pareto improvement.

**U:** no fresh replication, natural text, full Transformer, tied-head deployment, or capacity claim.

**Fact:** 12 serialized payloads reloaded; hashes and validation/test NLL replayed exactly (max difference 0); four existing tests pass. **Interpretation:** correcting task learnability exposes the direct-control byte equivalence. **Hypothesis:** a different TT-core coordinate or larger domain bank could change amortization, but this run gives no evidence for it.
