# MA-876 status

- Status: SCREENING
- Branch: `research/ma-876-shared-kv-translator-basis-20261111`
- Base commit: `df10d26a`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no; rank32 and prompt partitions frozen in FROZEN_SETTINGS.json
- Results committed: no
- Verification committed: no

## Next action

Run the three preregistered fresh worlds at frozen rank32; preserve all quality misses and replay metrics.

## Blockers

No GPU is present. CPU model weights and memory are available; this screen will report CPU timings and omit GPU claims.

## Decisions / rulings

- Pre-fresh amendment 02: exactly offset-aligned Wikitext test prompts for GPT-2→Pythia were insufficient for 64×3; use 16 disjoint test prompts per seed (48 total). This was decided before quality evaluation.

- Pre-fresh amendment 01: validation has only 45 aligned prompts for the GPT-2→Pythia pair, so fresh calibration uses disjoint 12-prompt chunks (36 total) while retaining 64 heldout test prompts per seed. This was set before model-quality development or fresh test selection.

- PA236 approximate cross-model transfer lane only; no exact-cache equivalence claim.
- The locally cached model weights are not copied to Git. Record snapshot IDs, file sizes, and hashes as provenance.
- Native PCA map coding is an exact representation control; any storage-quality result shared with it is non-Mirror-specific.
