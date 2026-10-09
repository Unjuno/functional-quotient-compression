# MA-469 — MEND-style gradient factors generate Mirror edit codes

Status: **FAIL for edit-bank compression**  
Evidence lane: MODEL EDITING / EDIT SUCCESS / LOCALITY / CODE BYTES / LATENCY  
Protocol frozen: `5f1f18c4`; fresh worlds 46910–46912.

## H — Hypothesis

Representing rank-one edit deltas with per-edit polar Mirror coordinates and one shared output basis will preserve edit success, paraphrase behavior, and locality while shrinking a MEND-style edit bank.

## T — Test

Analytic 2D linear edits to an identity base map. Each edit maps key `k` to `k+d`, yielding update `d kᵀ / ||k||²`. Compared MEND-style key+delta factors, Mirror magnitude/angle codes plus a shared identity basis, dense updates, and half-precision factor storage. Measured key success, perturbed-key behavior, unrelated-key locality drift, generation/apply wall time, and actual serialized N=1/20/64 payloads across three fresh worlds × three seeds.

## D — FAIL

**Fact:** At N=64, Mirror and MEND had effectively equal edit error (1.18e-7 vs 9.99e-8), paraphrase error (6.65e-8 vs 4.21e-8), and locality drift (5.61e-7 for both). Mirror payload was 3,105B (48.52B/edit), larger than MEND factors at 2,853B (44.58B/edit), and missed the <=80% storage gate. Dense updates cost 3,365B; half-precision factors cost 2,661B with edit error 1.86e-4. At N=20 Mirror happened to serialize slightly smaller than MEND, but the N=64 frontier reversed after charging the shared basis and metadata.

**Interpretation:** Polar Mirror coordinates are an exact re-encoding of the MEND output-delta vector here. They preserve locality but do not compress the edit bank; the simple half-precision factor control is smaller.

## C — Strongest counter-hypothesis

The editing update is already rank one. Storing its two factors is the natural minimum description, while polar coordinates add a shared basis and decode work without reducing per-edit information.

## U — Unknown

Nonlinear neural editing, learned MEND editors, factual paraphrases, and large-model locality remain untested. This is an analytic mechanism screen, not a MEND reproduction.
