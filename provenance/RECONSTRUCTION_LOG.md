# Reconstruction Log

This repository is reconstructed from three handoff packages supplied on 2026-09-02. The packages are treated as provenance sources, not as the desired repository layout.

## Source package fingerprints

| Source package | SHA-256 | Role in reconstruction |
|---|---|---|
| `vector_mirror_research_handoff_2026-09-02(4).zip` | `28b8dc38d162490a79c3c847d0085b7649053168c2770e5d1f9728c4c93e87a1` | compact theory/state summary; useful for cross-checking but not canonical text |
| `vector_mirror_research_handoff_2026-09-02_COMPLETE(1).zip` | `712cea162ae439950163a980d2c5af70d5ee63d1b86c4a78e4af240e422ef90b` | consolidated theory, E1–E7 exact experiment log, optimizer spec, legacy executable branch |
| `vector_mirror_handoff_2026-09-02_D120(1).zip` | `b442ffb4a1a52047b0ae097c50b89cf574b7be147a52d161bd6bb9e8fa156ecc` | D1–D120 provenance, claim boundaries, real-pilot contracts, synthetic scheduler/decision-geometry line |

The ZIP files themselves are **not copied into this repository**. Their scientific content is normalized into canonical documents/code while source hashes and mappings are retained here.

## Audit findings

### Compact handoff

- 15 entries, primarily Markdown/JSON state summaries.
- Four Markdown files contain ASCII control characters consistent with escaped TeX sequences being serialized incorrectly (`01_CORE_THEORY.md`, `03_ROOT_AND_CODEC_THEORY.md`, `04_VECTOR_OPTIMIZER.md`, `08_FORMULA_INDEX.md`).
- These files are evidence sources only; their text is not copied verbatim into canonical docs.

### COMPLETE handoff

- 38 entries including consolidated theory, E1–E7 exact experiment log, optimizer specification, prior reports, 8 Python scripts, and stored outputs.
- All 8 top-level Python scripts are syntactically valid.
- Two scripts reference missing absolute-path NPZ inputs: `mixed_gate_case.npz` and `mixed_gate_three_layer_case.npz`; those runs are therefore not fully reproducible from the package alone.
- The package SHA manifest validates its other listed files, but its self-hash entry is not a meaningful fixed-point integrity mechanism; canonical provenance instead uses source-package SHA plus git history.
- The E1–E7 exact results are valuable but need canonical executable reconstruction where original runners are incomplete/missing.

### D120 handoff

- 119 original checkpoint ZIPs are present for D1–D120; D96 is explicitly missing.
- Outer source manifest and checked nested manifests were internally consistent in audit.
- 94 Python-labeled files exist inside the checkpoint set: 53 compile, while 41 fail syntax compilation. The failing group is predominantly `*_reference.py` specification text with unterminated triple-quoted prose, not normal executable modules.
- These reference artifacts are to be reclassified as documentation/specification rather than copied into canonical `src/`.
- D56 and D57 contain the historical real-pilot / Transformer-extraction contract line that should be reconstructed as canonical contracts.
- D70–D120 are synthetic scheduler/decision-geometry evidence and must not be promoted to codec-rate or real-Transformer evidence.

## Reconstruction policy

For each source artifact, the canonical migration action is one of:

- `MERGE` — integrate scientifically valid content into a canonical document;
- `REIMPLEMENT` — create a clean executable implementation from a validated specification/log;
- `RECLASSIFY` — move pseudo-code/reference material to documentation/specification;
- `RETAIN_RAW` — retain necessary raw numerical data in a normalized experiment fixture when licensing/size permits;
- `RETRACT` — preserve the historical claim only in the claim/retraction record;
- `OMIT_DUPLICATE` — do not copy redundant source text;
- `MISSING` — record an unrecovered dependency/artifact explicitly.

A reconstruction never silently claims byte identity with the historic implementation. Recreated code is marked `RECONSTRUCTED` until an original source artifact is positively matched.

## Known missing / partial artifacts

- D96 original checkpoint ZIP.
- `mixed_gate_case.npz` referenced by the COMPLETE legacy branch.
- `mixed_gate_three_layer_case.npz` referenced by the COMPLETE legacy branch.
- canonical original runners for some logged E1–E7 exact experiments; these require reconstruction from logs/specifications unless recovered separately.

## Why the repository is not organized by ZIP or Delta

The canonical repository is organized by scientific object:

- theory / equivalence;
- codec / serializer;
- optimizer;
- experiments;
- evidence/claims;
- scheduler support.

Delta numbers and source ZIP paths remain provenance metadata only. This avoids turning a chronological handoff archive into the long-term software/research architecture.
