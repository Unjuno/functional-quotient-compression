# MA-517 — Function-vector composition with product coordinates

Status: SCREENING; development not run yet.
Branch: research/ma-517-function-vector-composition-20261008
Prior art: PA99, Function Vectors in Large Language Models.

## H — Hypothesis

On six inverse relation pairs in pinned Pythia-70m, multiplying shared-basis codes of the operand function vectors will produce the identity-composition behavior better than raw vector sum or difference on 48 held-out queries per world.

Mirror insertion: fit a centered function-vector basis on task IDs 0–11, code each function, and form an on-demand composition View as the elementwise product of the two operand codes followed by basis decoding. Tasks 12–15 are excluded from basis fitting. No composite activation vector is persistently cached.

## T — Frozen protocol

See PROTOCOL.json and freeze.json. The six inverse pairs are country/capital, state/postal-code, element/symbol, number/digit, English/French and English/Spanish. For each pair, both directions share the same underlying relation-entry support/query split; neither operand FV support contains a held-out composition path. Function vectors use the frozen MA-516 extraction implementation and pinned Pythia-70m checkpoint. Controls: no intervention, each operand alone, explicit sum, difference, raw Hadamard product, and identical native PCA-code product. Ranks {2,4,8,12}; dev seeds 51701/51702; fresh 51711–51713 sealed.

All operand vectors or code bases, codes, operator identifiers, pair IDs, model digest, layer and schema are charged in actual uncompressed NPZ bytes. Common model+tokenizer bytes are included in total deployment storage. The report separates support extraction, composition build, query scoring, and activation-operation proxy.

## D — Decision

Pending frozen development.

## C — Strongest counter-hypothesis

A product of function-vector coordinates is a hand-designed operator that may not correspond to sequential functional composition. Raw addition, subtraction, or multiplication can fail because they do not perform the underlying input-output transformation. If code product does help, ordinary PCA coordinate multiplication is the same method and is the direct attribution control.

## U — Scope limits

This is a constrained candidate-ranking screen on one 70M model and six inverse relation pairs. It is not broad reasoning, open-ended generation, or production model evidence.

## Evidence classification

- Facts: pending the frozen dev runs.
- Interpretation: the experiment distinguishes function-vector arithmetic from functional composition on known inverse pairs.
- Hypothesis: a structured product coordinate may represent useful composition beyond raw FV arithmetic; the native code-product control may fully explain it.
