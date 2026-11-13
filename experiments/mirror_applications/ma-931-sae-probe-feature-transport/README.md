# MA-931 — cross-model feature and probe transport

Status: **FAIL (scoped TinyStories screen)**.

Frozen amendment 01 used three pinned TinyStories GPT-Neo models and disjoint story rows (256:512 fit, 128:256 validation). It trained word-boundary and curated animal-word probes on target hidden states, then evaluated those fixed probes after independent affine or source × target × layer factorized bridges on held-out model directions. A shuffled-label probe was the negative control.

**Fact:** the factorized bridge retained 85.3% mean native AP for structural word-boundary labels but only 0.91% for the animal-word lexical labels. The independent affine bridge retained 97.8% structural AP but only 8.69% lexical AP. Full deployed factorized state was 237,834,895 B versus 238,729,489 B independent (0.375% smaller); connectors themselves were 37,953 B versus 932,547 B. Serialized bridge and probe replay had max absolute error 0.0.

**Interpretation:** factorization compressed connector state while losing tested feature information. The semantic-lexical task was badly transferred even by the independent affine control; the experiment does not establish that an affine bridge can generally transport semantics.

**Hypothesis:** feature family and model direction may determine what private connector state is needed. This narrow screen supports continued feature-specific auditing, not broad semantic-equivalence or Mirror-capacity claims.

Animal-word probing is a weak lexical concept test. This is not a PA242 reproduction, SAE generalization result, or end-to-end model-stitching benchmark. See [RESULTS.md](RESULTS.md), frozen [PROTOCOL.json](PROTOCOL.json), and [VERIFICATION.json](VERIFICATION.json).
