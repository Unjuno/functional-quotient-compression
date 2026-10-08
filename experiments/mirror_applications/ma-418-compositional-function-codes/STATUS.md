# MA-418 status: FAIL

## H — falsifiable hypothesis

Object and style Givens factors would compose on unseen pairs and approach a factorized latent decoder with no more than 1.1× bytes, while beating direct pair codes on held-out pairs.

## T — executed

Four object IDs × four style IDs, with diagonal pairs 00/11/22/33 held out from non-oracle training and development. Teacher functions compose separate object and style Givens rotations on one shared coordinate MLP. Compared shared, factorized Mirror, factorized ordinary latent codes, factorized FiLM, direct pair-table Givens codes with held-out entries untrained, and independent per-pair oracle decoders. 600 AdamW updates × batch 256; all methods selected LR 0.01 using seen-pair validation only; three fresh worlds × three seeds. Each run trained on 128 queries per seen pair and evaluated on disjoint 128 queries per pair. Actual serialized decoder/code payloads retained.

## D — FAIL

Fresh held-out-pair normalized RMSE: Mirror 0.0740; factorized latent 0.0327; FiLM 0.0149; direct pair table 0.2035; shared 0.1574; independent oracle 0.0128. Mirror's error is 2.26× factorized latent, missing the 1.1× gate, and 4.97× FiLM. Payload: Mirror 3,029B vs latent 3,541B (14.5% smaller), but quality fails. Direct pair table fails to extrapolate as expected; FiLM and factorized latent transfer better.

## C — strongest counter-hypothesis

Although the teacher is Givens-compositional, the learner's object/style factors may be weakly identifiable from only 12/16 combinations, and FiLM's more flexible activation parameters fit the shared decoder more easily. The result is consistent with a failure of Mirror coordinate optimization rather than a fundamental impossibility of composition.

## U — unresolved

More observed combinations, larger and more diverse object/style grids, near-convergence, natural visual/shape functions, and learned semantic factors remain untested. No natural compositionality claim is made.

