# MA-880 — Cross-model KV source × target factorization

Status: SCREENING. Prior art: PA236/PA237. The frozen protocol trains on a four-edge ordered-pair cycle, tunes CP rank on two other whole ordered pairs, and evaluates two additional unseen directions on fresh Wikitext-2 prompts. All four models are represented as both source and target in the training cycle.

## Hypothesis

A shared rank-32 map basis with compact source-model × target-model × normalized role factors can generate useful maps for unseen ordered pairs without pair-specific calibration. The success gates require quality close to full independent ridge and target-native prefill, as well as a 50% serialized-byte reduction.

## Mirror insertion and controls

The proposed coordinate is a CP product over source ID, target ID and normalized target layer/head/KV role, decoded through a shared map basis. Controls are native target prefill, independent weighted ridge fitted for each evaluated pair, and PCA map coding with pair-specific coefficients. CP factorization is itself an ordinary tensor-factorization baseline; a pass would not establish a Mirror-specific advantage.

See [PROTOCOL.json](PROTOCOL.json) for the frozen pair partitions, training recipe, prompt rules, gates and byte accounting. No fresh prompts are opened before development selects the CP rank.
