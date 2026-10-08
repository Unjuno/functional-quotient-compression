# MA-257 — Compositional Mirror context group

Status: **FAIL for Mirror-specific value at development; fresh remains sealed.**
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME
Base commit: `2be81372d70e1bdcaab706c932df66e17603b6d7`
Prior art: PA16 — Parameter Superposition.

## H — Hypothesis

Three small factorized Mirror coordinates can compose 512 logical linear functions from one physical operator and generalize to held-out factor combinations, with a useful byte/quality/compute frontier over ordinary free coefficient pairs. Independent task maps test where private parameters become necessary.

> **Mirror insertion:** this experiment adds factor coordinates `m=(m_a,m_b,m_c)` to one shared 6×6 linear operator so that task functions indexed by three factors compose without storing one full matrix per combination.

## T — Planned experiment

The aligned teacher applies three Givens rotations to disjoint two-dimensional planes of one shared matrix. Eight values per factor yield 512 task combinations. Factor-code fitting sees only combinations selected by a fixed residue split; held-out combinations test compositional generalization. A separate independent-matrix condition tests misalignment.

Controls are shared-only tying, native random-sign Parameter Superposition, native PA16 factorized rotational contexts (an exact functional control), factorized two-coefficient sine/cosine codebooks at FP16 and FP32, per-seen-task angle triplets, and independent full matrices. The independent upper is explicitly sample-unmatched on held-out combinations. Every method gets the same three supplied factor labels. Actual serialized payload bytes, fit work, inference work, and wall-clock are reported separately.

The development seeds selected support 1/8 (64 of 512 combinations) using validation only. In every development seed/condition/support, FP16 Mirror and the native PA16 rotational-context control had identical task errors and actual payload bytes. This exact prior-art equivalence fails the Mirror-specific gate, so locked fresh seeds remain unopened. This is a zero-optimizer post-fit screen, not learning efficiency or capacity evidence.

## D — Decision

**FAIL for Mirror-specific value at development; fresh sealed.** This is a preregistered direct-control failure, not a failed aligned mechanism. Three development worlds show the compositional aligned function family is representable by 24 factor angles and generalizes to held-out combinations from 64/512 support tasks. However, the closest native PA16 factorized rotational context uses the exact same angular code, function, payload length, and transform semantics. Measured CPU throughput varied around the same path and did not establish a runtime advantage. No fresh replication is needed to decide this identity claim; the fresh split remains unopened.

### Selected-support development facts

| Method | Actual payload | Mean validation nMSE | Mean test nMSE | Mean inference examples/s |
|---|---:|---:|---:|---:|
| Shared-only | 231 B | 0.17584 | 0.17594 | 50.3M |
| Mirror FP16 angles | 296 B | 6.60e-9 | 6.58e-9 | 5.67M |
| Native PA16 FP16 rotational contexts | 296 B | 6.60e-9 | 6.58e-9 | 6.03M |
| Factor coefficients FP16 | 348 B | 2.89e-8 | 2.89e-8 | 5.29M |
| Mirror FP32 angles | 344 B | 7.69e-16 | 7.67e-16 | 5.67M |
| Factor coefficients FP32 | 444 B | 9.01e-16 | 9.00e-16 | 5.34M |
| Per-task FP16 angles, seen tasks only | 3,848 B | 6.55e-9 | 6.64e-9 | 5.41M |
| Random-sign PSP | 263 B | 384.59 | 382.83 | 4.85M |
| Independent full oracle | 73,824 B | 6.63e-16 | 6.62e-16 | 50.6M |

The independent upper directly calibrates all 512 task maps, unlike the held-out-composition models. On independent task maps, Mirror FP16 mean test nMSE was 8.64, while coefficient FP16 was 1.036 and independent full was near zero; this marks the private-capacity boundary. Random-sign PSP had mean nMSE 382.8 in the aligned condition. All reported bytes are serialized payloads, including schema, method tags, factors, codes, matrices, and context seed where relevant.

## Fact / Interpretation / Hypothesis

**Fact:** development seeds 25701–25703 and support fractions 1/8, 2/8, 3/8 were evaluated. Support 1/8 was selected by the frozen validation rule. Across all 18 condition/seed/support matches, FP16 Mirror and native PA16 factorized rotational context had identical validation/test metrics and payload lengths. The selected support payloads replay exactly. Seven tests pass; no fresh seed was opened.

**Interpretation:** M3-style compositional function generation works for the intentionally aligned commuting-orbit teacher, but the extra code is precisely the rotational context already present in PA16; this screen establishes no Mirror-specific advantage. The FP16 angle format also beats FP16 free coefficient pairs on this teacher, but the native rotational context is the stronger explanation.

**Hypothesis:** the useful extension is to test rotational context composition on naturally learned task functions or a different strong native method; the current PA16 crossover does not justify a Mirror-specific claim.

## C — Strongest counter-hypothesis

PA16 already studies compositional rotational contexts. The native factorized rotational context is an exact function and code alias of Mirror in this experiment; it has equal bytes, so no Mirror-specific advantage is possible here.

## U — Unresolved

Natural task functions, trained representations, continual learning, language-model quality, and near-converged fixed-byte capacity are outside this screen. Fresh seeds 25711–25713 remain sealed. The independent upper uses more direct task examples than compositional methods.
