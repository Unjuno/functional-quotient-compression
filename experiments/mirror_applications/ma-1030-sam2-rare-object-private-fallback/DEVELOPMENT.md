# MA-1030 development record

Development world seed 103000; only the preregistered fallback threshold was selected. No audit-world files were generated or opened.

| Method / threshold | J&F | object-memory bytes | ID switches | private selections |
|---|---:|---:|---:|---|
| independent | 0.592189 | 786,432 | 0 | none |
| hard_shared | 0.011912 | 196,608 | 0 | none |
| mirror_givens | 0.010071 | 196,800 | 11 | none |
| additive_rank1 | 0.296495 | 393,312 | 12 | none |
| mirror_private_fallback / 0.15 | 0.186918 | 393,456 | 14 | f0→42,f1→42,f2→42,f3→42,f4→42,f5→42 |
| mirror_private_fallback / 0.30 | 0.160211 | 360,688 | 15 | f0→42,f1→42,f2→42,f3→42,f5→42 |
| mirror_private_fallback / 0.50 | 0.120654 | 262,384 | 18 | f0→42,f4→42 |
| mirror_private_fallback / 0.80 | 0.010071 | 196,848 | 11 | none |

Frozen threshold: **0.15**, selected by the preregistered maximum-J&F rule (development J&F 0.186918); byte count breaks ties.

Observed development fact: at the selected threshold the residual-energy fallback allocated its one private slot on all six yielded frames, selecting object ID 42 each time; the registered shifted-appearance object is ID 99. This is retained as a development warning, not used to change the selector or threshold after selection. The audit now tests this frozen rule unchanged.

The runner commit after protocol freeze only adds persistence of development rows and shared-cache bytes; selector, score, threshold grid, gates, and generated images are unchanged.
