# MA-931 results

## H — hypothesis

A compact source × target × layer factorized residual-stream bridge can transport both structural and semantic-lexical probe features across held-out TinyStories models, preserving at least 90% of native AP while using at most 25% of independent affine connector bytes. Feature families may behave differently; the animal lexical probe is only a narrow label probe.

## T — trial

Three pinned GPT-Neo TinyStories models (1M/3M/8M; 8 layers) ran on CPU. Connector fit used 256 stories (train rows 256:512); evaluation used 128 held-out validation stories (rows 128:256). Three seeds (93101, 93107, 93113), rank 8, layers 4/8, fit directions 0↔1 and 1↔2, evaluated held-out directions 0↔2 and 2↔0. The same target probe scored native target activations and bridged source activations. Controls were independent affine connector, native target representation, and prevalence-matched shuffled-label probe. There were 46 training and 26 validation animal-word positives, above frozen minimums.

## D — FAIL (scoped)

### Fact

| Feature | Condition | Mean AP | Mean balanced accuracy | Mean AP retained vs native |
|---|---:|---:|---:|---:|
| Structural word boundary | Native | 0.9930 | 0.9081 | 1.0000 |
| Structural word boundary | Factorized bridge | 0.8472 | 0.4811 | 0.8531 |
| Structural word boundary | Independent affine | 0.9716 | 0.7253 | 0.9784 |
| Structural word boundary | Shuffled negative | 0.9130 | 0.5798 | 0.9194 |
| Animal lexical | Native | 0.7501 | 0.9755 | 1.0000 |
| Animal lexical | Factorized bridge | 0.00683 | 0.4519 | 0.00912 |
| Animal lexical | Independent affine | 0.0671 | 0.6322 | 0.0869 |
| Animal lexical | Shuffled negative | 0.00762 | 0.4852 | 0.0104 |

Factorized connector serialized size was 37,953 B; independent connector was 932,547 B (95.93% connector reduction). The saved connector bytes did not materially shrink full deployment: 237,834,895 B factorized versus 238,729,489 B independent, a 0.375% reduction, because model weights and probe state dominate. Probe payload was 10,781 B and metadata 436 B, charged in both. Factorized connector, independent connector, and probe reload replay max absolute output error was 0.0. Connector MAC proxy was 2,688–2,880 factorized vs 16,448–16,640 independent. The script reports shared fit/evaluation wall attribution; no serving-throughput claim is made.

### Interpretation

The factorized connector did not preserve both tested probe families and missed the frozen gate. Structural AP retention was below the 90% signal criterion; lexical AP retention was below the 75% failure threshold. Independent affine improved structural transfer but also failed to preserve the lexical probe. Thus this run identifies a representation/family mismatch, not a Mirror-specific semantic transport win. Connector compression alone is not a useful full-model storage frontier improvement here.

### Hypothesis

Feature-specific connectors or private residual state may be necessary for some feature families and directions. That remains untested. The unexpectedly low AP for animal words after independent transport could reflect cross-model representation mismatch, the narrow lexical concept, or limitations in the linear probe setup.

## C — strongest counter-hypothesis

The animal-word concept is rare and lexically narrow, and the target probe may rely on model-specific token/representation geometry. Since the independent affine control also retained only 8.69% of native AP, the result cannot attribute the lexical failure to factorization or Mirror coordinates alone. Structural labels may be encoded in low-level token-boundary artifacts and are not evidence of semantic feature sharing.

## U — unconfirmed

No official PA242 datasets/SAE banks/steering interventions were reproduced. We did not test broad semantics, causal interventions, end-to-end quality, GPU inference, or a private feature residual. These measures are a scoped probe screen, not near-converged capacity evidence.
